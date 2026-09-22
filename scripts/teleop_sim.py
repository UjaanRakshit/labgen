"""Drive one or two simulated YAMs from pose-streaming devices, in real time.

Runs inside WSL, in the Isaac Lab environment. Connects OUT to the controller
running on Windows (see scripts/teleop_touch.py for why that direction).

    python teleop_sim.py bench_arm.usda /path/to/yam.urdf            # one arm
    python teleop_sim.py bench_arm.usda /path/to/yam.urdf --bimanual # two

Nothing below knows what is on the other end of the wire. A touch pad, a WebXR
phone and a Quest controller are all just `labgen.devices.PoseEvent` streams,
and each event carries the hand it belongs to, so two devices in one session
drive two arms without this loop caring which is which.

THE SECOND ARM'S BASE OFFSET IS NOT MEASURED. It is a labelled placeholder, and
`labgen.control.BimanualRig` reports the rig unverified because of it. That is
deliberate: the offset between two bases is the one number deciding whether
every two-handed reach is right, it cannot be derived from anything in this
repo, and a guess produces a rig that looks correct and is wrong by however far
the guess was off. Recording refuses while it stands.

THE GRASP GEOMETRY IS ALSO UNVERIFIED -- the finger pads are measured off the
URDF mesh, not off real hardware with calipers. Same rule: fine to drive, not
fine to record training data from.
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
from arm_drive import configure_real_drives, gravity_torques            # noqa: E402
from grasp import set_state, GraspFK, N_ARM, use_pad_colliders, with_fingers, enable_pad_torsion         # noqa: E402

sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import (YAM_JAWS, YAM_PAD, ArmSpec, BimanualRig,      # noqa: E402
                            solve_pose, yam_arm_spec)
from labgen.hardware import YAM_V1                                         # noqa: E402
from labgen.devices import (DEFAULT_PORT, RelativeRetargeter,             # noqa: E402
                            TcpPoseSource, Workspace)

# Real time is arithmetic: each control tick must advance SUBSTEPS * DT of
# simulated time in under 1/CONTROL_HZ of wall time. 4 substeps at 1/240 was
# exactly real time on paper and 0.74x in practice, because one solver step
# costs ~5.6 ms of wall clock for 4.17 ms of sim. A bigger step costs about the
# same per step, so the fix is fewer, larger steps.
DT = 1.0 / 120.0
SUBSTEPS = 2
CONTROL_HZ = 60.0

# Contact parameters stay at the importer defaults here, NOT the ones
# labgen.settle ships. Those give a 2.5 ms constraint time constant, which
# MuJoCo resolves only at dt <= 1.25 ms, and this loop runs at 8.3 ms to stay
# real time. The cost is a few millimetres of resting penetration: visible,
# tolerable while driving, not acceptable for recorded demonstrations.

# Contact buffer size. The default is sized for the single-arm scene; a second
# arm overflows it, and MuJoCo says so once and then DROPS contacts:
#
#     narrowphase overflow - please increase nconmax to 67
#
# Dropped contacts are not a slowdown, they are missing forces: bodies
# interpenetrate unopposed and the scene explodes some minutes later with no
# input at all. That is what "diverged -- stopping" was. Sized well above the
# observed peak so a third arm or a busier bench does not silently reintroduce
# it.
NCONMAX = 512
NJMAX = 2048

import os as _os
DEBUG_LAG = _os.environ.get("LABGEN_DEBUG_LAG", "") not in ("", "0")

DOWN = np.array([0.0, 0.0, -1.0])
LOOK_AT = np.array([0.05, 0.35, 0.12])
EYE = LOOK_AT + np.array([1.05, 0.05, 0.70])

# Warm-started IK: teleop moves the target a millimetre or two per tick, so the
# previous configuration is an excellent seed. Iteration count from
# scripts/probe_ikrate.py (FK 0.17 ms, Jacobian 2.08 ms), not from taste.
IK_ITERS = 3
IK_TOL_M = 0.002

# Reachable band for a top-down wrist, measured (scripts/probe_reach.py), and
# expressed RELATIVE TO EACH ARM'S OWN BASE so a second arm gets the same
# envelope around itself rather than inheriting the first arm's.
REACH_BOX_LOCAL = ((-0.42, 0.10, 0.02), (0.42, 0.52, 0.25))
READY_LOCAL = np.array([0.20, 0.28, 0.22])

# The second arm FACES the first, across the bench. That orientation is
# sourced, if weakly: the lab's own teleop config says "two arms facing each
# other do not share one [reset pose]" (pairlab/yam_teleop/deployment/
# config.yaml). An earlier placeholder had the arms side by side facing the
# same way, which contradicts that. The DISTANCE is still unmeasured: 0.70 m
# across puts the bench objects (y 0.28-0.42 m from the first arm) roughly
# between the two bases. PLACEHOLDER, and the rig reports unverified.
PLACEHOLDER_OFFSET_M = (0.0, 0.70, 0.0)
PLACEHOLDER_YAW_WXYZ = (0.0, 0.0, 0.0, 1.0)      # 180 deg about z: facing arm_left
PLACEHOLDER_SOURCE = (
    "PLACEHOLDER 700 mm across the bench, facing the first arm. Facing: from "
    "yam_teleop deployment/config.yaml ('two arms facing each other'). "
    "Distance: NOT a measurement of the real rig."
)


def default_rig(bimanual: bool, lower, upper) -> BimanualRig:
    arms = [yam_arm_spec(lower, upper, name="arm_left")]
    if bimanual:
        arms.append(yam_arm_spec(lower, upper, name="arm_right",
                                 base_position_m=PLACEHOLDER_OFFSET_M,
                                 base_orientation_wxyz=PLACEHOLDER_YAW_WXYZ,
                                 placement_source=PLACEHOLDER_SOURCE))
    return BimanualRig(arms)


def aim(viewer, eye, target):
    """Point the camera. Without this the viewer opens looking at nothing,
    which is indistinguishable from a broken renderer."""
    d = np.asarray(target, float) - np.asarray(eye, float)
    viewer.set_camera(tuple(float(v) for v in eye),
                      math.degrees(math.atan2(d[2], float(np.hypot(d[0], d[1])))),
                      math.degrees(math.atan2(d[1], d[0])))


class ArmInstance:
    """One arm inside the combined model: its slices, limits, solver state.

    Everything the control loop needs to drive one arm without knowing how many
    others exist. Two YAMs differ only in name, base transform, and which slice
    of the joint arrays they own.
    """

    def __init__(self, spec: ArmSpec, builder_arm, dofs, coords, lower, upper):
        self.spec, self.name = spec, spec.name
        self.dofs, self.coords = dofs, coords
        self.lo, self.hi = lower[:N_ARM], upper[:N_ARM]
        self.masses = np.asarray(builder_arm.body_mass, float)
        self.axes = np.asarray(builder_arm.joint_axis, float)
        self.coms = np.asarray(builder_arm.body_com, float)
        self.base = np.asarray(spec.base_position_m, float)
        # Reach box and ready pose are defined in the arm's OWN frame, so a
        # rotated base must rotate them, not merely translate them -- otherwise
        # an arm facing the other way is handed a workspace behind itself.
        w, x, y, z = spec.base_orientation_wxyz
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        corners = np.array([[a, b, c] for a in (REACH_BOX_LOCAL[0][0], REACH_BOX_LOCAL[1][0])
                            for b in (REACH_BOX_LOCAL[0][1], REACH_BOX_LOCAL[1][1])
                            for c in (REACH_BOX_LOCAL[0][2], REACH_BOX_LOCAL[1][2])])
        world = corners @ R.T + self.base
        self.workspace = Workspace(lower_m=tuple(world.min(axis=0)),
                                   upper_m=tuple(world.max(axis=0)))
        self.ready = self.base + R @ READY_LOCAL
        self.body_offset = 0
        self.fk = None
        self.q_cmd = None
        self.retarget = None
        self.ik_fail = 0
        self.ik_ms = 0.0
        self.solves = 0
        self.track_m = 0.0
        self.target_m = None      # last commanded hand position
        self.phys_m = 0.0         # physical lag, summed
        self.phys_peak_m = 0.0
        self.phys_n = 0


def build(scene_usda: Path, urdf: Path, rig: BimanualRig, collide: bool = True,
          contact: tuple[float, float] | None = None, finger_force_n: float | None = None,
          pad_torsion_m: float | None = None):
    """Scene plus one arm per rig entry, each at its own base transform.

    `contact=(ke, kd)` overrides every shape's contact parameters. Teleop leaves
    it None to stay real time at dt=1/120; anything whose numbers must be
    physically faithful -- the grasp acceptance test -- passes labgen.settle's
    (CONTACT_KE, CONTACT_KD) and runs at a dt that satisfies timeconst >= 2*dt.

    `finger_force_n` sets the per-finger force limit, so a grip-force sweep can
    rebuild with a different ceiling without editing the drive defaults.

    `pad_torsion_m` gives the pads torsional friction (grasp.enable_pad_torsion);
    None leaves Newton's condim=3, under which a rim-held beaker pivots freely
    about the grip axis.
    """
    builder = newton.ModelBuilder()
    builder.add_usd(str(scene_usda))

    instances = []
    for i, spec in enumerate(rig.arms):
        sc, sd = builder.joint_coord_count, builder.joint_dof_count
        arm = newton.ModelBuilder()
        w, x, y, z = spec.base_orientation_wxyz
        arm.add_urdf(str(urdf),
                     xform=wp.transform(wp.vec3(*spec.base_position_m),
                                        wp.quat(x, y, z, w)),
                     floating=False, enable_self_collisions=False,
                     collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                     parse_visuals_as_colliders=collide, mesh_maxhullvert=64)
        lower = np.asarray(arm.joint_limit_lower, float)
        upper = np.asarray(arm.joint_limit_upper, float)
        if collide:
            # The fingertip hull spans +/-36 mm and engulfs anything between the
            # jaws; measured, it ejects a 250 mL beaker 9.7 m. Authored pads
            # instead.
            use_pad_colliders(arm, YAM_PAD, verbose=(i == 0))
        dofs = slice(sd, sd + arm.joint_dof_count)
        coords = slice(sc, sc + arm.joint_coord_count)
        # MERGE FIRST, then configure. configure_drives clamps its slice to the
        # builder's current dof count, so calling it before add_builder gives an
        # empty range and silently applies NO gains -- both arms then have no
        # position control at all and simply fall over. It fails quietly
        # because clamping a slice is not an error.
        builder.add_builder(arm)
        # The REAL robot's per-joint actuators (labgen.hardware.YAM_V1, from the
        # i2rt 1.1.2 config the rig runs): kp 80/10, kd 5/1.5, peak 28/10 N.m,
        # Coulomb friction 0.3/0.06 N.m. Replaces kp=3000 on every joint, which
        # was chosen to make the arm hold still and hid a broken gravity
        # feedforward the whole time it was in use.
        configure_real_drives(builder, YAM_V1, dof_slice=dofs, n_fingers=2,
                              verbose=(i == 0))
        instances.append(ArmInstance(spec, arm, dofs, coords, lower, upper))

    if contact is not None:
        ke, kd = contact
        for i in range(builder.shape_count):
            builder.shape_material_ke[i] = ke
            builder.shape_material_kd[i] = kd
    if finger_force_n is not None:
        for inst in instances:
            start, stop, _ = inst.dofs.indices(builder.joint_dof_count)
            for d in range(stop - 2, stop):
                builder.joint_effort_limit[d] = finger_force_n

    if pad_torsion_m is not None:
        if not collide:
            raise ValueError("pad torsion needs the pad colliders (collide=True)")
        enable_pad_torsion(builder, pad_torsion_m)

    model = builder.finalize()
    # Arms were appended in order, so each owns a contiguous run of bodies at
    # the end. Walk backwards to hand each one its body offset.
    end = model.body_count
    for inst in reversed(instances):
        end -= len(inst.masses)
        inst.body_offset = end
    return model, instances


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scene_usda", type=Path)
    ap.add_argument("urdf", type=Path)
    ap.add_argument("--bimanual", action="store_true",
                    help="two arms. The second base offset is a PLACEHOLDER "
                         "until measured, and the rig reports unverified.")
    ap.add_argument("--host", default="AUTO")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--stream", action="store_true",
                    help="render headless and push JPEG frames to the "
                         "controller, which serves them to any browser. More "
                         "reliable than the WSLg window (software COPY MODE, "
                         "~40 ms a frame) and it puts the view on the phone.")
    ap.add_argument("--frame-port", type=int, default=9872)
    ap.add_argument("--stream-quality", type=int, default=70)
    ap.add_argument("--view-width", type=int, default=960)
    ap.add_argument("--view-height", type=int, default=540)
    ap.add_argument("--no-collide", action="store_true",
                    help="import the arms with NO collision geometry. The YAM "
                         "URDF ships zero <collision> elements, so colliders "
                         "come from the visual meshes; without them the arm "
                         "renders perfectly and passes through everything.")
    ap.add_argument("--rotation", action="store_true",
                    help="let device orientation steer the approach axis. Off "
                         "by default: an identity device orientation maps to "
                         "approach +Z, straight UP, which this arm cannot "
                         "reach over a bench, so every solve fails and the arm "
                         "looks frozen.")
    ap.add_argument("--seconds", type=float, default=0.0)
    args = ap.parse_args()

    # Joint limits belong to whichever URDF revision is actually loaded, so read
    # them once rather than keeping a second copy in this file to drift.
    probe = newton.ModelBuilder()
    probe.add_urdf(str(args.urdf),
                   xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                   floating=False, enable_self_collisions=False,
                   collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                   parse_visuals_as_colliders=False)
    lower = np.asarray(probe.joint_limit_lower, float)
    upper = np.asarray(probe.joint_limit_upper, float)

    rig = default_rig(args.bimanual, lower, upper)
    print(f"rig: {len(rig)} arm(s) -- {', '.join(rig.names)}")
    if not rig.verified:
        print("   " + rig.why_unverified().replace(chr(10), chr(10) + "   "))
        print("   -> driving is fine; RECORDING IS REFUSED while this stands.")
    if not YAM_PAD.verified:
        print(f"   finger pads UNVERIFIED: {YAM_PAD.source}")

    host = (TcpPoseSource.wsl_default_gateway() if args.host == "AUTO"
            else args.host)
    print(f"connecting to the controller at {host}:{args.port} ...")
    source = TcpPoseSource(host, args.port)
    print("connected.")

    model, arms = build(args.scene_usda, args.urdf, rig,
                        collide=not args.no_collide)
    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20,
                                         nconmax=NCONMAX, njmax=NJMAX)

    # Drag axes align to the CAMERA, not the robot base: "drag up the screen"
    # has to mean "push away from the viewer" or the operator does mental
    # arithmetic on every motion.
    fwd = LOOK_AT - EYE
    fwd[2] = 0.0
    fwd /= np.linalg.norm(fwd)
    right = np.array([fwd[1], -fwd[0], 0.0])
    screen_frame = np.column_stack([right, fwd, np.array([0.0, 0.0, 1.0])])

    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()

    for inst in arms:
        inst.fk = GraspFK(model, inst.coords,
                          body_offset=inst.body_offset,
                          n_bodies=len(inst.masses), pad=YAM_PAD)
        seed = solve_pose(inst.fk, inst.ready, inst.lo, inst.hi, approach=DOWN)
        if not seed.ok:
            print(f"{inst.name}: cannot reach its ready pose {inst.ready}: {seed}")
            return 1
        inst.q_cmd = seed.q
        inst.retarget = RelativeRetargeter(position_scale=args.scale,
                                           use_orientation=args.rotation,
                                           frame=screen_frame,
                                           workspace=inst.workspace)
        q_all[inst.coords] = with_fingers(inst.q_cmd, YAM_JAWS.q_open)
        reached, _ = inst.fk.pose(inst.q_cmd)
        print(f"   {inst.name}: base {tuple(np.round(inst.base, 3))} "
              f"ready {np.round(reached, 3)}  {seed}")

    set_state(model, s0, q_all)
    set_state(model, s1, q_all)

    frame_sock = None
    if args.stream:
        frame_sock = socket.create_connection((host, args.frame_port), timeout=10.0)
        frame_sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"streaming frames to {host}:{args.frame_port}")

    viewer = None
    if args.stream or not args.headless:
        viewer = ViewerGL(width=args.view_width, height=args.view_height,
                          headless=bool(args.stream), vsync=False)
        viewer.set_model(model)
        aim(viewer, EYE, LOOK_AT)

    # Command the ready pose BEFORE the loop. joint_target_q is zeros out of the
    # box, and zeros is a fully extended arm: until the operator first engaged,
    # the drive pulled the arm flat onto the bench while the IK sat holding a
    # perfectly good configuration it had never been asked to command.
    targets = control.joint_target_q.numpy().copy()
    ff = control.joint_f.numpy().copy()
    for inst in arms:
        targets[inst.coords] = with_fingers(inst.q_cmd, YAM_JAWS.q_open)
    control.joint_target_q.assign(targets)

    # Which arm a device drives. A single device saying hand="right" on a
    # one-arm rig still has to land somewhere, so unknown hands fall back.
    by_hand = {}
    for i, inst in enumerate(arms):
        for key in (inst.name, inst.name.replace("arm_", ""), str(i)):
            by_hand.setdefault(key, inst)
    default_arm = arms[0]

    period = 1.0 / CONTROL_HZ
    next_tick = time.perf_counter()
    frames = last_frames = 0
    phys_ms = grav_ms = view_ms = 0.0
    started = last_report = time.perf_counter()
    print(f"ready. Engage on the controller to drive "
          f"{'an arm (pick LEFT/RIGHT)' if len(arms) > 1 else 'the arm'}.")

    try:
        while True:
            if args.seconds and time.perf_counter() - started > args.seconds:
                break

            latest = source.poll_latest()
            if source.closed:
                print("controller closed the connection")
                return 1

            if latest is not None:
                inst = by_hand.get(latest.hand, default_arm)
                hand_now, _ = inst.fk.pose(inst.q_cmd)
                target = inst.retarget.update(latest, hand_now)
                if target is not None:
                    t0 = time.perf_counter()
                    res = solve_pose(inst.fk, target.position_m, inst.lo, inst.hi,
                                     approach=(target.approach if args.rotation
                                               else DOWN),
                                     seed=inst.q_cmd, restarts=1,
                                     iters=IK_ITERS, tol_m=IK_TOL_M)
                    inst.ik_ms += (time.perf_counter() - t0) * 1000
                    inst.solves += 1
                    inst.q_cmd = res.q
                    if not res.ok:
                        inst.ik_fail += 1
                    reached, _ = inst.fk.pose(inst.q_cmd)
                    inst.track_m += float(np.linalg.norm(reached - target.position_m))
                    inst.target_m = np.asarray(target.position_m, float).copy()
                    gap = (1.0 - float(np.clip(latest.grip, 0.0, 1.0))) \
                        * YAM_JAWS.max_gap_m
                    targets[inst.coords] = with_fingers(inst.q_cmd,
                                                        YAM_JAWS.q_for_gap(gap))
                    control.joint_target_q.assign(targets)

            # Gravity feedforward per arm, from the COMMANDED configuration.
            # Without it the drive spends 81% of its rated torque holding
            # station and saturates on the first millimetre of lag.
            t_g = time.perf_counter()
            for inst in arms:
                bw = inst.fk._eval(inst.q_cmd)
                ff[inst.dofs.start: inst.dofs.start + N_ARM] = -gravity_torques(
                    bw, inst.masses, inst.axes, coms=inst.coms,
                    offset=inst.body_offset,
                    n_arm=N_ARM)
            control.joint_f.assign(ff.astype(np.float32))
            grav_ms += (time.perf_counter() - t_g) * 1000

            t_p = time.perf_counter()
            for _ in range(SUBSTEPS):
                contacts = model.collide(s0)
                solver.step(s0, s1, control, contacts, DT)
                s0, s1 = s1, s0
            phys_ms += (time.perf_counter() - t_p) * 1000

            bq = s0.body_q.numpy()
            if not np.isfinite(bq).all():
                print("diverged -- stopping")
                return 1

            # PHYSICAL tracking: where the simulated hand actually is, against
            # where it was told to go. The IK figure above only says the solver
            # found a configuration; with the real robot's soft gains the arm
            # lags that configuration, and the lag is what an operator feels and
            # what sim-to-real has to match.
            for inst in arms:
                if inst.target_m is not None:
                    lag = float(np.linalg.norm(
                        inst.fk.grasp_point_from_state(bq) - inst.target_m))
                    inst.phys_m += lag
                    inst.phys_peak_m = max(inst.phys_peak_m, lag)
                    inst.phys_n += 1
                    if DEBUG_LAG and inst.phys_n <= 40:
                        hand = inst.fk.grasp_point_from_state(bq)
                        print(f"   [lag] tick {inst.phys_n:3d} lag {lag*1000:7.1f} mm  "
                              f"target {np.round(inst.target_m, 3)}  "
                              f"hand {np.round(hand, 3)}", flush=True)

            t_v = time.perf_counter()
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
                            frame_sock.sendall(len(jpeg).to_bytes(4, "big") + jpeg)
                        except OSError:
                            print("frame stream closed")
                            frame_sock = None
            view_ms += (time.perf_counter() - t_v) * 1000

            frames += 1
            now = time.perf_counter()
            if now - last_report > 2.0:
                n = max(frames - last_frames, 1)
                per_arm = "  ".join(
                    f"{a.name.replace('arm_', '')}: ik {a.track_m / max(a.solves, 1) * 1000:4.2f}mm "
                    f"phys {a.phys_m / max(a.phys_n, 1) * 1000:5.1f}mm "
                    f"(pk {a.phys_peak_m * 1000:5.1f}) {a.ik_fail}bad" for a in arms)
                print(f"   {n / (now - last_report):5.1f} Hz | physics "
                      f"{phys_ms / n:5.2f} | grav {grav_ms / n:4.2f} | view "
                      f"{view_ms / n:5.2f} ms | {per_arm}", flush=True)
                last_report, last_frames = now, frames
                phys_ms = grav_ms = view_ms = 0.0

            next_tick += period
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_tick = time.perf_counter()
    except KeyboardInterrupt:
        print("\nstopped by operator")
    finally:
        source.close()
        if viewer is not None:
            viewer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
