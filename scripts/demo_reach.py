"""The demo: YAM arm on the labgen bench, reaching a catalog keypoint.

Loads the SAME `bench_5.usda` the physics tests run against, adds the YAM from
its MJCF, solves IK to the beaker's `rim_grasp` keypoint as declared in
`catalog.py`, and renders the motion.

Scripted motion, not a policy. It shows that the generated scene and the robot
share one consistent coordinate frame, and that a catalog keypoint is a usable
target. It shows nothing about whether a policy can do a task.

    python demo_reach.py <bench.usda> <yam.xml> <out_dir> [--seconds N]
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton
from newton.viewer import ViewerGL

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasp import set_state                                    # noqa: E402
from arm_drive import configure_drives                      # noqa: E402
from arm_ik import ArmFK, lerp_path, solve_ik                # noqa: E402

# labgen itself, straight off the Windows mount -- the catalog and the scene
# spec are the authority for where things are, not numbers retyped here.
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG                           # noqa: E402
from labgen.types import SceneSpec                           # noqa: E402

WIDTH, HEIGHT = 1600, 900
FPS = 60


def aim(viewer, eye, target) -> None:
    d = np.asarray(target, float) - np.asarray(eye, float)
    viewer.set_camera(
        tuple(float(v) for v in eye),
        math.degrees(math.atan2(d[2], float(np.hypot(d[0], d[1])))),
        math.degrees(math.atan2(d[1], d[0])),
    )


def keypoint_world(scene: SceneSpec, instance_id: str, keypoint: str) -> np.ndarray:
    """A catalog keypoint in the robot base frame.

    All bench_5 objects are axis-aligned (identity quaternion), so this is a
    translation. A rotated object would need the quaternion applied; that is
    deliberately not faked here, it would just be wrong silently.
    """
    obj = scene.by_id(instance_id)
    if obj.orientation_wxyz != (1.0, 0.0, 0.0, 0.0):
        raise SystemExit(f"{instance_id} is rotated; keypoint_world does not handle that yet")
    item = CATALOG[obj.catalog_key]
    return np.asarray(obj.position) + np.asarray(item.keypoints[keypoint])


def main() -> int:
    argv = sys.argv
    bench = Path(argv[1])
    arm_src = Path(argv[2])
    out_dir = Path(argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)

    scene = SceneSpec.read(
        f"/mnt/c/Ujaan Docx/Research/labgen/examples/{bench.stem}.json")

    builder = newton.ModelBuilder()
    builder.add_usd(str(bench))
    scene_dof = builder.joint_dof_count
    scene_coord = builder.joint_coord_count

    arm = newton.ModelBuilder()
    arm.add_mjcf(
        str(arm_src),
        xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
        floating=False,
        enable_self_collisions=False,
        collapse_fixed_joints=True,
        up_axis=newton.Axis.Z,
    )
    arm_dof = arm.joint_dof_count
    arm_coord = arm.joint_coord_count
    lower = np.asarray(arm.joint_limit_lower, dtype=float)[:arm_dof]
    upper = np.asarray(arm.joint_limit_upper, dtype=float)[:arm_dof]
    builder.add_builder(arm)

    # Newton keeps two different indexings and they do not line up here.
    # A free body costs 7 COORDS (position + quaternion) but 6 DOF, so the
    # bench's four free bodies occupy 28 coords and 24 dof. joint_q and
    # control.joint_target_q are coord-space; the drive gain arrays are
    # dof-space. Slicing one with the other's index writes into the wrong
    # joints entirely -- it put the IK 200 mm off and looked like the target
    # was out of reach.
    dofs = slice(scene_dof, scene_dof + arm_dof)
    coords = slice(scene_coord, scene_coord + arm_coord)
    configure_drives(builder, dof_slice=dofs)

    model = builder.finalize()
    print(f"scene+arm: {model.body_count} bodies, {model.shape_count} shapes; "
          f"arm dof [{dofs.start}:{dofs.stop}] coord [{coords.start}:{coords.stop}]")

    # --- solve the reach -------------------------------------------------
    fk = ArmFK(model, coords)
    home = np.zeros(arm_dof)
    print(f"home gripper: {np.round(fk.position(home), 4)}")

    # A "ready" pose to start and end from. The YAM's documented home is all
    # six joints at zero, and at that configuration its elbow sits 0.24 m
    # BEHIND the base (link3 at x = -0.244) with the whole arm lying flat along
    # the worktop. That is the correct home pose and it is a terrible opening
    # frame -- it reads as a collapsed arm. Real arms have a ready posture
    # distinct from their zero, so this solves for one: gripper up and slightly
    # forward, clear of everything on the bench.
    ready_target = np.array([0.18, 0.16, 0.42])
    q_ready, e0, ok0 = solve_ik(fk, ready_target, lower, upper, seed=home)
    print(f"IK ready   : residual {e0 * 1000:6.2f} mm  converged={ok0}")

    # Visit each object's grasp keypoint in turn, straight from the catalog.
    # Nothing here knows an object's dimensions -- only its keypoint name --
    # which is the property that lets a 250 mL beaker be swapped for a 500 mL
    # one without touching the motion.
    tour = [("beaker", "rim_grasp"), ("test_tube", "neck_grasp"),
            ("petri", "rim_grasp")]

    waypoints, durations, labels = [q_ready], [], ["ready"]
    seed = q_ready
    for instance, kp in tour:
        target = keypoint_world(scene, instance, kp)
        above = target + np.array([0.0, 0.0, 0.12])
        q_above, ea, oka = solve_ik(fk, above, lower, upper, seed=seed)
        q_at, eb, okb = solve_ik(fk, target, lower, upper, seed=q_above)
        print(f"IK {instance:10} {kp:11} above {ea * 1000:5.2f} mm ({oka})  "
              f"at {eb * 1000:5.2f} mm ({okb})   target {np.round(target, 3)}")
        waypoints += [q_above, q_at, q_at, q_above]
        durations += [1.3, 0.7, 0.5, 0.7]
        labels += [f"{instance}:above", f"{instance}:{kp}", "hold", "retreat"]
        seed = q_above
    waypoints.append(q_ready)
    durations.append(1.3)
    labels.append("ready")

    hold_start, hold_end = 0.8, 1.2
    path = ([q_ready] * int(hold_start * FPS)
            + lerp_path(waypoints, durations, FPS)
            + [q_ready] * int(hold_end * FPS))
    print(f"trajectory: {len(path)} frames ({len(path) / FPS:.1f} s), "
          f"{len(tour)} keypoints visited")

    # --- simulate and render ---------------------------------------------
    state_0, state_1 = model.state(), model.state()
    control = model.control()
    # Start the arm AT the ready pose rather than letting it swing there from
    # zero on the first frame.
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = q_ready
    model.joint_q.assign(q_all.astype(np.float32))
    set_state(model, state_0, model.joint_q.numpy())
    set_state(model, state_1, model.joint_q.numpy())

    solver = newton.solvers.SolverMuJoCo(model, iterations=100, ls_iterations=50)
    viewer = ViewerGL(width=WIDTH, height=HEIGHT, headless=True, vsync=False)
    viewer.set_model(model)

    targets = control.joint_target_q.numpy().copy()
    look_at = np.array([0.0, 0.24, 0.14])
    dt = 1.0 / FPS

    from PIL import Image

    written = []
    for i, q in enumerate(path):
        targets[coords] = q
        control.joint_target_q.assign(targets)

        contacts = model.collide(state_0)
        solver.step(state_0, state_1, control, contacts, dt)
        state_0, state_1 = state_1, state_0
        if not np.isfinite(state_0.body_q.numpy()).all():
            print(f"   diverged at frame {i}")
            break

        # Slow orbit so the depth of the scene reads on a flat video.
        a = -1.05 + 0.45 * math.sin(2.0 * math.pi * i / len(path))
        eye = look_at + np.array([1.05 * math.cos(a), 1.05 * math.sin(a), 0.62])
        aim(viewer, eye, look_at)

        viewer.begin_frame(i * dt)
        viewer.log_state(state_0)
        viewer.end_frame()

        frame = viewer.get_frame()
        if frame is None:
            print("   no pixels")
            break
        arr = frame.numpy() if hasattr(frame, "numpy") else np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        # NOT flipped. ViewerGL.get_frame() already returns rows top-down.
        # An np.flipud() here mirrored every render vertically: the bench
        # filled the top of frame with the background below it, the arm
        # appeared to hang downward, and a box on the worktop read as a hole
        # punched into it. Everything looked plausible enough to not question.
        arr = arr[:, :, :3]

        p = out_dir / f"frame_{i:04d}.png"
        Image.fromarray(arr).save(p)
        written.append(p)
        if i % 60 == 0:
            print(f"   frame {i}/{len(path)}")

    viewer.close()

    # Where did the gripper actually end up, and did the glassware stay put?
    final = state_0.body_q.numpy()
    print(f"\nfinal gripper: {np.round(final[-1, :3], 4)}")
    print(f"wrote {len(written)} frames to {out_dir}")
    return 0 if written else 1


if __name__ == "__main__":
    sys.exit(main())
