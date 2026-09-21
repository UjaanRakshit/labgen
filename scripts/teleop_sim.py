"""Drive the simulated arm from a pose-streaming device, in real time.

Runs inside WSL, in the Isaac Lab environment. Connects OUT to the bridge
running on Windows (see scripts/teleop_bridge.py for why that direction).

    python teleop_sim.py bench_arm.usda /path/to/yam.urdf [--host AUTO]

The device is whatever is on the other end of the bridge. Nothing below knows
whether that is a phone or a Quest controller, which is the entire point of
`labgen.devices.PoseSource`: the Quest plugs into this same slot with no change
here.

THE GRASP IS STILL KINEMATIC. Contact grasping is item D and is not done. An
operator can fly the arm around and pick things up in this viewer, and those
motions are real enough to be worth watching, but a demonstration recorded
through this path must NOT be used to train anything -- it would teach a policy
grasps that do not survive contact. That is why this script does not record.
"""

from __future__ import annotations

import argparse
import io
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp
from PIL import Image

import newton
from newton.viewer import ViewerGL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_drive import (DEFAULT_KD, DEFAULT_KE, YAM_RATED_EFFORT_NM,       # noqa: E402
                       configure_drives, gravity_torques)
from grasp import (GraspFK, N_ARM, use_pad_colliders,                     # noqa: E402
                   with_fingers)

sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS, YAM_PAD, solve_pose                           # noqa: E402
from labgen.devices import (DEFAULT_PORT, RelativeRetargeter,             # noqa: E402
                            TcpPoseSource, Workspace)

# Real time is the constraint, and it is arithmetic: each control tick must
# advance SUBSTEPS * DT of simulated time in less than 1/CONTROL_HZ of wall
# time. 4 substeps at 1/240 was exactly real time on paper and 0.74x in
# practice -- one solver step costs ~5.6 ms of wall clock for 4.17 ms of sim.
# A bigger step costs about the same per step, so the fix is fewer, larger
# steps rather than a faster solver.
DT = 1.0 / 120.0
SUBSTEPS = 2
CONTROL_HZ = 60.0

# Contact parameters are deliberately left at the importer defaults here, not
# set to the ones labgen.settle ships. Those give a 2.5 ms constraint time
# constant, which MuJoCo can only resolve at dt <= 1.25 ms, and this loop runs
# at 8.3 ms to stay real time. The cost is a few millimetres of resting
# penetration -- visible, tolerable while the grasp is kinematic anyway, and
# NOT acceptable for recorded demonstrations. See labgen/settle.py.

# Straight down. The gripper approaches a bench from above, and that is the
# band the IK actually solves in.
DOWN = np.array([0.0, 0.0, -1.0])

# Middle of the bench, in the robot base frame.
LOOK_AT = np.array([-0.02, 0.30, 0.10])
EYE = LOOK_AT + np.array([0.75, -0.55, 0.55])

# Where the arm parks before the operator engages.
READY_POSE = np.array([0.20, 0.28, 0.22])

# Warm-started IK. Teleop moves the target a millimetre or two per tick, so the
# previous configuration is an excellent seed and the solver needs very few
# iterations -- unlike planning, which starts cold with multiple restarts.
# The iteration count comes from scripts/probe_ikrate.py, not from taste.
IK_ITERS = 3
IK_TOL_M = 0.002

# Measured reachable band for a top-down wrist (scripts/probe_reach.py): solves
# at radius 30-46 cm up to z=250 mm, and not above that at this radius. The
# operator is clamped into it rather than allowed to command poses the IK will
# silently fail to reach, which on screen looks like the arm ignoring them.
WORKSPACE = Workspace(lower_m=(-0.42, 0.10, 0.02), upper_m=(0.42, 0.52, 0.25))


def aim(viewer, eye, target):
    """Point the camera. Without this the viewer opens looking at nothing.

    Newton's ViewerGL does not frame the scene for you, and an unset camera
    shows an empty window that reads exactly like a broken renderer. That cost
    a session here.
    """
    d = np.asarray(target, float) - np.asarray(eye, float)
    viewer.set_camera(tuple(float(v) for v in eye),
                      math.degrees(math.atan2(d[2], float(np.hypot(d[0], d[1])))),
                      math.degrees(math.atan2(d[1], d[0])))


def build(scene_usda: Path, urdf: Path, collide: bool = True):
    builder = newton.ModelBuilder()
    builder.add_usd(str(scene_usda))
    sc, sd = builder.joint_coord_count, builder.joint_dof_count

    arm = newton.ModelBuilder()
    arm.add_urdf(str(urdf), xform=wp.transform(wp.vec3(0.0, 0.0, 0.0),
                                               wp.quat_identity()),
                 floating=False, enable_self_collisions=False,
                 collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                 parse_visuals_as_colliders=collide, mesh_maxhullvert=64)
    lower = np.asarray(arm.joint_limit_lower, float)
    upper = np.asarray(arm.joint_limit_upper, float)
    # Swap the fingertip hulls for authored pad boxes BEFORE merging. A convex
    # hull of the L-shaped tip spans +/-36 mm and engulfs anything between the
    # jaws; measured, it ejects a 250 mL beaker 9.7 m.
    if collide:
        use_pad_colliders(arm, YAM_PAD)

    builder.add_builder(arm)

    dofs = slice(sd, sd + arm.joint_dof_count)
    coords = slice(sc, sc + arm.joint_coord_count)
    configure_drives(builder, dof_slice=dofs, n_fingers=2,
                     effort=YAM_RATED_EFFORT_NM, verbose=False)
    model = builder.finalize()
    return model, arm, dofs, coords, lower, upper


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_usda", type=Path)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("--host", default="AUTO",
                    help="bridge host; AUTO finds the Windows host from WSL")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--scale", type=float, default=1.0,
                    help="phone centimetres to hand centimetres")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--no-collide", action="store_true",
                    help="import the arm with NO collision geometry. The YAM "
                         "URDF ships zero <collision> elements, so colliders "
                         "have to come from the visual meshes; without them "
                         "the arm renders perfectly and passes through "
                         "everything. On by default because an arm that "
                         "touches nothing is not worth driving.")
    ap.add_argument("--view-width", type=int, default=960)
    ap.add_argument("--view-height", type=int, default=540)
    ap.add_argument("--stream", action="store_true",
                    help="render headless and push JPEG frames to the "
                         "controller, which serves them to any browser. "
                         "More reliable than the WSLg window, which runs "
                         "in software COPY MODE at ~40 ms a frame, and it "
                         "puts the view on the phone as well as the PC.")
    ap.add_argument("--frame-port", type=int, default=9872)
    ap.add_argument("--stream-quality", type=int, default=70)
    ap.add_argument("--seconds", type=float, default=0.0,
                    help="stop cleanly after N seconds; 0 runs until Ctrl-C. "
                         "Needed because SIGTERM skips the summary.")
    ap.add_argument("--rotation", action="store_true",
                    help="let the phone's orientation steer the approach axis. "
                         "Off by default: an identity device orientation maps to "
                         "approach +Z, i.e. straight UP, which this arm cannot "
                         "reach over a bench -- every solve then fails and the "
                         "arm looks frozen. Position-only with a fixed downward "
                         "approach is the useful first version.")
    args = ap.parse_args()

    host = (TcpPoseSource.wsl_default_gateway() if args.host == "AUTO"
            else args.host)
    print(f"connecting to the bridge at {host}:{args.port} ...")
    source = TcpPoseSource(host, args.port)
    print("connected.")

    model, arm, dofs, coords, lower, upper = build(args.scene_usda, args.urdf,
                                                   collide=not args.no_collide)
    fk = GraspFK(model, coords)
    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20)

    arm_masses = np.asarray(arm.body_mass, float)
    arm_axes = np.asarray(arm.joint_axis, float)
    body_offset = model.body_count - len(arm_masses)

    lo, hi = lower[:N_ARM], upper[:N_ARM]

    # Start at a real ready pose, not the midpoint of the joint limits.
    #
    # The midpoint is a folded configuration whose hand sits OUTSIDE the
    # workspace box below, so every commanded target was clamped to the
    # boundary the instant the operator engaged: measured as a target box of
    # 0.0 x 45.1 mm for a commanded 60 x 60 mm circle. The arm tracked that
    # faithfully, which is exactly why it was hard to see -- tracking error
    # stayed at 1.4 mm the whole time. The hand was following the wrong target.
    READY = READY_POSE
    assert WORKSPACE.contains(READY), "the ready pose must be inside the workspace"
    seed = solve_pose(fk, READY, lo, hi, approach=DOWN)
    if not seed.ok:
        print(f"could not reach the ready pose: {seed}")
        return 1
    q_cmd = seed.q
    start, _ = fk.pose(q_cmd)
    print(f"ready pose {np.round(start, 4)} (asked for {READY}), {seed}")

    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = with_fingers(q_cmd, YAM_JAWS.q_open)
    model.joint_q.assign(q_all.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)

    frame_sock = None
    if args.stream:
        frame_sock = socket.create_connection((host, args.frame_port), timeout=5.0)
        frame_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"streaming frames to {host}:{args.frame_port}")

    viewer = None
    if args.stream:
        # Headless renderer + JPEG push. The window is not opened at all.
        viewer = ViewerGL(width=args.view_width, height=args.view_height,
                          headless=True, vsync=False)
        viewer.set_model(model)
        aim(viewer, EYE, LOOK_AT)
    elif not args.headless:
        # 960x540, not 720p. Measured through WSLg the viewer costs ~37 ms a
        # frame at 1280x720 and drags a 50 Hz loop down to 18 Hz -- the window
        # is more expensive than the physics and the IK combined. Resolution is
        # the cheapest lever on that.
        viewer = ViewerGL(width=args.view_width, height=args.view_height,
                          headless=False, vsync=False)
        viewer.set_model(model)
        # Same viewpoint the task demo renders from: over the operator's right
        # shoulder, looking down at the middle of the bench.
        aim(viewer, EYE, LOOK_AT)

    # Drag axes are aligned to the CAMERA, not to the robot base. The camera
    # looks along LOOK_AT - EYE, so "drag up the screen" has to mean "push away
    # from the viewer" or the operator is doing mental arithmetic on every
    # motion. Screen right and screen forward, both flattened into the bench
    # plane and made orthonormal.
    fwd = LOOK_AT - EYE
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0], 0.0])
    screen_frame = np.column_stack([right, fwd, np.array([0.0, 0.0, 1.0])])

    retarget = RelativeRetargeter(position_scale=args.scale,
                                  frame=screen_frame,
                                  use_orientation=args.rotation,
                                  workspace=WORKSPACE)
    targets = control.joint_target_q.numpy().copy()
    ff = control.joint_f.numpy().copy()

    # Command the ready pose BEFORE the loop starts. joint_target_q is zeros
    # out of the box, and zeros is a fully extended arm: until the operator
    # first engaged, the drive was pulling the arm flat onto the bench while
    # the IK sat holding a perfectly good ready configuration it had never
    # been asked to command.
    targets[coords] = with_fingers(q_cmd, YAM_JAWS.q_open)
    control.joint_target_q.assign(targets)

    period = 1.0 / CONTROL_HZ
    next_tick = time.perf_counter()
    frames = 0
    engaged_frames = 0
    ik_fail = 0
    ik_ms_total = phys_ms_total = grav_ms_total = view_ms_total = 0.0
    last_frames = 0
    track_m_total, track_n = 0.0, 0
    hand_path: list = []
    target_path: list = []
    last_report = time.perf_counter()
    print("ready. Press and hold MOVE on the phone to engage the arm.")

    started = time.perf_counter()
    try:
        while True:
            if args.seconds and time.perf_counter() - started > args.seconds:
                break
            # Drain to the NEWEST sample. A control loop that works through a
            # backlog is an operator watching the arm replay where the phone
            # used to be; only the freshest pose means anything.
            latest = source.poll_latest()
            if source.closed:
                print("bridge closed the connection")
                return 1

            if latest is not None:
                hand_now, _ = fk.pose(q_cmd)
                target = retarget.update(latest, hand_now)
                if target is not None:
                    engaged_frames += 1
                    t0 = time.perf_counter()
                    res = solve_pose(fk, target.position_m, lo, hi,
                                     approach=(target.approach if args.rotation
                                               else DOWN), seed=q_cmd,
                                     restarts=1, iters=IK_ITERS, tol_m=IK_TOL_M)
                    ik_ms_total += (time.perf_counter() - t0) * 1000
                    q_cmd = res.q
                    if not res.ok:
                        ik_fail += 1
                    # Does the hand actually go where it was told? The frame
                    # rate says the loop is keeping up; this says the loop is
                    # CORRECT, and those are different questions.
                    reached, _ = fk.pose(q_cmd)
                    track_m_total += float(np.linalg.norm(reached - target.position_m))
                    track_n += 1
                    hand_path.append(reached.copy())
                    target_path.append(np.asarray(target.position_m).copy())

                    gap = (1.0 - float(np.clip(latest.grip, 0.0, 1.0))) * YAM_JAWS.max_gap_m
                    targets[coords] = with_fingers(q_cmd, YAM_JAWS.q_for_gap(gap))
                    control.joint_target_q.assign(targets)

            # Gravity feedforward from the COMMANDED configuration. Without it
            # the drive spends 81% of its rated torque holding station and
            # saturates on the first millimetre of lag; measured, see
            # arm_drive.YAM_RATED_EFFORT_NM.
            t_grav = time.perf_counter()
            bw = fk._eval(q_cmd)
            ff[dofs.start: dofs.start + N_ARM] = -gravity_torques(
                bw, arm_masses, arm_axes, offset=body_offset, n_arm=N_ARM)
            control.joint_f.assign(ff.astype(np.float32))
            grav_ms_total += (time.perf_counter() - t_grav) * 1000

            t_phys = time.perf_counter()
            for _ in range(SUBSTEPS):
                contacts = model.collide(s0)
                solver.step(s0, s1, control, contacts, DT)
                s0, s1 = s1, s0

            phys_ms_total += (time.perf_counter() - t_phys) * 1000

            if not np.isfinite(s0.body_q.numpy()).all():
                print("diverged -- stopping")
                return 1

            t_view = time.perf_counter()
            if viewer is not None:
                aim(viewer, EYE, LOOK_AT)
                viewer.begin_frame(frames / CONTROL_HZ)
                viewer.log_state(s0)
                viewer.end_frame()
                if frame_sock is not None:
                    f = viewer.get_frame()
                    if f is not None:
                        arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
                        if arr.dtype != np.uint8:
                            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
                        buf = io.BytesIO()
                        Image.fromarray(arr[:, :, :3]).save(
                            buf, format="JPEG", quality=args.stream_quality)
                        jpeg = buf.getvalue()
                        try:
                            frame_sock.sendall(
                                len(jpeg).to_bytes(4, "big") + jpeg)
                        except OSError:
                            print("frame stream closed")
                            frame_sock = None
            view_ms_total += (time.perf_counter() - t_view) * 1000

            frames += 1
            now = time.perf_counter()
            if now - last_report > 2.0:
                n = max(frames - last_frames, 1)
                rate = n / (now - last_report)
                print(f"   {rate:5.1f} Hz | IK {ik_ms_total / max(engaged_frames, 1):5.2f} "
                      f"| physics {phys_ms_total / n:5.2f} "
                      f"| grav {grav_ms_total / n:5.2f} "
                      f"| view {view_ms_total / n:5.2f} ms "
                      f"| tracking {track_m_total / max(track_n, 1) * 1000:5.2f} mm "
                      f"| {ik_fail} IK unconverged, {retarget.jumps_rejected} dropouts, "
                      f"{source.malformed} malformed", flush=True)
                last_report, last_frames = now, frames
                phys_ms_total = grav_ms_total = view_ms_total = 0.0

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped by operator")
    finally:
        if len(hand_path) > 60:
            cut = len(hand_path) // 3
            h = np.array(hand_path[cut:])
            t = np.array(target_path[cut:])
            hs = (h.max(axis=0) - h.min(axis=0)) * 1000
            ts = (t.max(axis=0) - t.min(axis=0)) * 1000
            print(f"target box {ts[0]:6.1f} x {ts[1]:6.1f} x {ts[2]:5.1f} mm")
            print(f"hand   box {hs[0]:6.1f} x {hs[1]:6.1f} x {hs[2]:5.1f} mm "
                  f"over {len(h)} engaged frames")
            print(f"worst |hand-target| {np.linalg.norm(h - t, axis=1).max()*1000:.2f} mm")
        source.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
