"""The settle test, on real physics.

CLAUDE.md calls this the cheapest high-value check in the project: step the
scene 2 seconds with no actuation; nothing moves more than 2 mm and nothing
explodes.

It is also the only thing that can actually answer whether the SDF colliders
work. `Mesh.sdf` reads None after import because Newton builds SDFs deferred,
so inspecting the model proves nothing either way. If the SDF is absent or too
coarse to resolve a 1 mm wall, the thin-walled objects are the ones that will
show it -- they sink, jitter, or launch.

Run with Isaac Lab's venv:
    ~/isaac/IsaacLab/.venv/bin/python settle_test.py bench_5.usda
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import newton

USDA = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_5.usda")
DURATION = 2.0          # seconds, per CLAUDE.md
TOLERANCE = 0.002       # metres, per CLAUDE.md
FPS = 240               # solver substep rate


def main() -> int:
    builder = newton.ModelBuilder()
    result = builder.add_usd(str(USDA))
    model = builder.finalize()

    body_paths = {v: k for k, v in result["path_body_map"].items()}
    names = [body_paths.get(i, f"body{i}").replace("/World/", "")
             for i in range(model.body_count)]

    print(f"bodies: {model.body_count}  shapes: {model.shape_count}")
    print(f"gravity: {model.gravity}")

    state_0 = model.state()
    state_1 = model.state()
    control = model.control()

    start = state_0.body_q.numpy().copy()

    solver = newton.solvers.SolverMuJoCo(model)
    print(f"solver: {type(solver).__name__}")

    dt = 1.0 / FPS
    steps = int(DURATION * FPS)
    print(f"stepping {steps} x {dt * 1000:.2f} ms = {DURATION:.1f} s, no actuation\n")

    peak = np.zeros(model.body_count)
    exploded = False
    for i in range(steps):
        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        q = state_0.body_q.numpy()
        if not np.isfinite(q).all():
            print(f"  !! non-finite pose at step {i} (t={i * dt:.3f}s) -- EXPLODED")
            exploded = True
            break
        peak = np.maximum(peak, np.linalg.norm(q[:, :3] - start[:, :3], axis=1))

    end = state_0.body_q.numpy()

    print(f"{'body':14} {'start z':>9} {'end z':>9} {'dz':>9} {'peak |d|':>10}  verdict")
    print("-" * 70)
    worst = 0.0
    failures = []
    for i, name in enumerate(names):
        dz = end[i, 2] - start[i, 2]
        d = float(np.linalg.norm(end[i, :3] - start[i, :3]))
        worst = max(worst, d)
        ok = d <= TOLERANCE and np.isfinite(d)
        if not ok:
            failures.append(name)
        print(f"{name:14} {start[i, 2]:9.5f} {end[i, 2]:9.5f} {dz:+9.5f} "
              f"{peak[i]:10.5f}  {'ok' if ok else 'MOVED'}")

    print()
    print(f"worst displacement: {worst * 1000:.3f} mm   tolerance: {TOLERANCE * 1000:.1f} mm")
    if exploded:
        print("RESULT: FAIL -- simulation diverged")
        return 1
    if failures:
        print(f"RESULT: FAIL -- {', '.join(failures)} moved beyond tolerance")
        return 1
    print("RESULT: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
