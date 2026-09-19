"""bench_5 with the YAM arm standing at the robot base frame origin.

This is the demo view of T4: the arm exists, it is anchored at the origin of
`SceneSpec.robot_base_frame`, and it does not intersect the bench. No actuator
tuning, no policy, no task -- an arm that exists and does not fall over is the
whole of the task, and anything past that belongs to a later phase.

The arm is driven by a scripted joint trajectory, NOT a policy. That
distinction matters and should survive into whatever gets shown: this
demonstrates that the scene and the robot coexist in one consistent frame, and
it demonstrates nothing at all about whether a policy can do a task.

    python bench_with_arm.py <bench.usda> <yam.urdf> [out_dir] [--seconds N]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

import newton
from newton.viewer import ViewerGL

WIDTH, HEIGHT = 1600, 900
FPS = 60


def aim(viewer, eye, target) -> None:
    d = np.asarray(target, float) - np.asarray(eye, float)
    yaw = math.degrees(math.atan2(d[1], d[0]))
    pitch = math.degrees(math.atan2(d[2], float(np.hypot(d[0], d[1]))))
    viewer.set_camera(tuple(float(v) for v in eye), pitch, yaw)


def main() -> int:
    argv = sys.argv
    bench = Path(argv[1])
    urdf = Path(argv[2])
    out_dir = Path(argv[3] if len(argv) > 3 and not argv[3].startswith("-") else "frames_arm")
    seconds = float(argv[argv.index("--seconds") + 1]) if "--seconds" in argv else 6.0
    out_dir.mkdir(parents=True, exist_ok=True)

    builder = newton.ModelBuilder()

    # The bench first, so its object indices match the standalone scene.
    builder.add_usd(str(bench))
    bench_shapes = builder.shape_count
    print(f"bench: {builder.body_count} bodies, {bench_shapes} shapes")

    # The arm at the origin of the robot base frame. bench_5 is authored so
    # that z = 0 IS the bench top surface, which is where the arm bolts down --
    # so the identity transform here is the claim the scene was built to make.
    import warp as wp

    arm = newton.ModelBuilder()

    # Position drives with real gains BEFORE importing, so every joint the URDF
    # creates inherits them. Without this the joints are free hinges: the arm
    # has no way to hold itself up, folds onto the bench under gravity, and the
    # solver then diverges on the resulting interpenetration. That looked like a
    # physics failure and was actually an unconfigured actuator.
    #
    # The YAM's own actuatorfrcrange is +/-10 N.m, so the effort limit is the
    # robot's, not ours. CLAUDE.md says not to tune actuator gains in T4;
    # these are the minimum needed to make the arm stand, not a tuned set, and
    # they are flagged as such.
    arm.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
        target_ke=400.0,        # NOT tuned -- enough to hold against gravity
        target_kd=40.0,
        effort_limit=10.0,      # from the YAM spec: actuatorfrcrange +/-10 N.m
        armature=0.01,
    )

    arm.add_urdf(
        str(urdf),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,                 # bolted to the bench, not free-flying
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        up_axis=newton.Axis.Z,
    )
    print(f"arm: {arm.body_count} bodies, {arm.shape_count} shapes, "
          f"{arm.joint_count} joints, dof={arm.joint_dof_count}")
    builder.add_builder(arm)

    model = builder.finalize()
    print(f"combined: {model.body_count} bodies, {model.shape_count} shapes")

    state_0, state_1 = model.state(), model.state()
    control = model.control()

    # Forward kinematics from the joint configuration onto the body transforms.
    # Newton leaves body_q at identity otherwise, so the arm renders folded into
    # the origin and snaps into its real pose on step 1. That reads as an arm
    # collapsing under gravity; it is actually an uninitialised state.
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, state_1)
    # The default iteration budget warns "solver iterations limit reached" on
    # this scene; contact between an articulated arm and five free bodies needs
    # more than a bare bench does.
    solver = newton.solvers.SolverMuJoCo(model, iterations=100, ls_iterations=50)

    viewer = ViewerGL(width=WIDTH, height=HEIGHT, headless=True, vsync=False)
    viewer.set_model(model)

    # Frame the working area: the arm at the origin plus the objects it would
    # reach, not the whole 1.5 m bench. The far bench edge dominating the shot
    # is what the first pass looked like.
    look_at = (-0.05, 0.24, 0.10)
    dt = 1.0 / FPS
    frames = int(seconds * FPS)

    # The model's dof count includes the bench's free bodies (6 each), so the
    # arm's joints are the LAST arm_dof entries. Driving from index 0 would
    # push on the beaker's free-body dofs instead of the shoulder, which looks
    # like a physics explosion and is actually an indexing bug.
    ndof = model.joint_dof_count
    arm_dof = arm.joint_dof_count
    base = ndof - arm_dof
    print(f"model dof={ndof}; arm occupies [{base}:{ndof}] ({arm_dof} dof)")

    target_attr = "joint_target_q" if hasattr(control, "joint_target_q") else "joint_target"
    buf = getattr(control, target_attr, None)
    targets = np.zeros(ndof, dtype=np.float32) if buf is None else buf.numpy().copy()
    print(f"driving control.{target_attr}")

    written = []
    for i in range(frames):
        t = i * dt

        # A slow scripted sweep. Amplitudes are small on purpose: this is a
        # "the arm is here and behaves" demo, not a reach.
        if buf is not None and arm_dof:
            # Hold the home pose for the first second so the arm is visibly
            # stable before anything moves -- that stillness IS T4's acceptance
            # criterion, and a demo that starts mid-swing hides it.
            # Hold the home pose. T4's acceptance is "the arm spawns at the
            # robot base frame origin and does not intersect the bench", and
            # holding still is how you show that.
            #
            # An earlier version swept the joints on a sine. It looked like a
            # demo and was a lie: joint2/joint3 cannot go negative, so the sweep
            # drove the arm down into the worktop and scattered the glassware.
            # A task motion needs IK against the catalog keypoints, which is
            # later work -- not a waveform.
            targets[base:base + arm_dof] = 0.0
            buf.assign(targets)

        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0

        if not np.isfinite(state_0.body_q.numpy()).all():
            print(f"   diverged at frame {i}")
            break

        angle = -1.35 + 0.9 * (i / max(frames - 1, 1))
        eye = (look_at[0] + 1.15 * math.cos(angle),
               look_at[1] + 1.15 * math.sin(angle),
               look_at[2] + 0.72)
        aim(viewer, eye, look_at)

        viewer.begin_frame(t)
        viewer.log_state(state_0)
        viewer.end_frame()

        frame = viewer.get_frame()
        if frame is None:
            print("   no pixels")
            break
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

        p = out_dir / f"frame_{i:04d}.png"
        Image.fromarray(arr).save(p)
        written.append(p)
        if i % 60 == 0:
            print(f"   frame {i}/{frames}")

    viewer.close()
    print(f"wrote {len(written)} frames")

    if written:
        from PIL import Image

        imgs = [Image.open(p) for p in written]
        gif = out_dir.parent / "bench_with_arm.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:],
                     duration=int(1000 / FPS), loop=0, optimize=True)
        imgs[len(imgs) // 3].save(out_dir.parent / "bench_with_arm.png")
        print(f"wrote {gif}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
