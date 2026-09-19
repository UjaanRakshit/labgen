"""Render a labgen scene to PNG frames (and a GIF/MP4 if tooling allows).

Headless OpenGL through Newton's own viewer, so this runs over SSH or in WSL
with no display. It is deliberately the same load path as settle_test.py -- the
picture is of the thing that actually simulates, not of a separate render-only
asset. A demo built from a different asset than the one under test is how you
end up showing a bench that does not exist.

    python render_scene.py <scene.usda> [out_dir] [--seconds N] [--spin]
"""

from __future__ import annotations

import inspect
import math
import sys
from pathlib import Path

import numpy as np

import newton
from newton.viewer import ViewerGL

WIDTH, HEIGHT = 1600, 900
FPS = 60


def parse_args(argv: list[str]):
    usda = Path(argv[1] if len(argv) > 1 else "bench_5.usda")
    out = Path(argv[2] if len(argv) > 2 and not argv[2].startswith("-") else "frames")
    seconds = 4.0
    if "--seconds" in argv:
        seconds = float(argv[argv.index("--seconds") + 1])
    return usda, out, seconds, "--spin" in argv


def aim(viewer, eye, target) -> None:
    """Point the camera at `target` from `eye`.

    ViewerGL takes (pos, pitch, yaw) in degrees rather than a look-at, so the
    angles are derived here. Keeping the call site in look-at terms matters:
    the camera pose has to be reproducible against a real photograph later
    (phase 1's definition of done is a side-by-side), and "eye here, looking
    there" is what a tripod position actually gives you.
    """
    d = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    horizontal = float(np.hypot(d[0], d[1]))
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], horizontal))
    viewer.set_camera(tuple(float(v) for v in eye), pitch, yaw)


def main() -> int:
    usda, out_dir, seconds, spin = parse_args(sys.argv)
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = newton.ModelBuilder()
    builder.add_usd(str(usda))
    model = builder.finalize()
    print(f"loaded {usda}: {model.body_count} bodies, {model.shape_count} shapes")

    state_0, state_1 = model.state(), model.state()
    control = model.control()
    solver = newton.solvers.SolverMuJoCo(model)

    viewer = ViewerGL(width=WIDTH, height=HEIGHT, headless=True, vsync=False)
    viewer.set_model(model)

    # The bench sits in front of the robot base, which is the origin. Look at
    # the middle of the worktop from slightly above and to the operator's right
    # -- roughly where a person standing at the bench would see it from.
    look_at = (0.0, 0.28, 0.05)
    radius, height_ = 0.62, 0.42

    dt = 1.0 / FPS
    frames = int(seconds * FPS)
    print(f"rendering {frames} frames at {WIDTH}x{HEIGHT} ...")

    written = []
    for i in range(frames):
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        angle = (-0.9 + (2.0 * math.pi * i / frames if spin else 0.0))
        eye = (look_at[0] + radius * math.cos(angle),
               look_at[1] + radius * math.sin(angle),
               look_at[2] + height_)
        aim(viewer, eye, look_at)

        viewer.begin_frame(i * dt)
        viewer.log_state(state_0)
        viewer.end_frame()

        frame = viewer.get_frame()
        if frame is None:
            print("   get_frame() returned None -- no pixels available")
            break
        # warp array -> numpy; np.asarray() on a wp.array slices instead.
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        # NOT flipped. ViewerGL.get_frame() already returns rows top-down.
        # An np.flipud() here mirrored every render vertically: the bench
        # filled the top of frame with the background below it, the arm
        # appeared to hang downward, and a box on the worktop read as a hole
        # punched into it. Everything looked plausible enough to not question.
        arr = arr

        from PIL import Image

        path = out_dir / f"frame_{i:04d}.png"
        Image.fromarray(arr).save(path)
        written.append(path)
        if i % 30 == 0:
            print(f"   frame {i}/{frames}")

    viewer.close()
    print(f"\nwrote {len(written)} PNG frames to {out_dir}/")

    if written:
        from PIL import Image

        imgs = [Image.open(p) for p in written]
        gif = out_dir.parent / f"{usda.stem}.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, optimize=True)
        print(f"wrote {gif} ({gif.stat().st_size / 1024:.0f} KB)")

        still = out_dir.parent / f"{usda.stem}.png"
        imgs[-1].save(still)
        print(f"wrote {still}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
