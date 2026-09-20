"""Multiple pick-and-place tasks on the labgen bench.

Each task is expressed against catalog keypoints by NAME -- `body_grasp`,
`plate_center` -- never against coordinates. Nothing here knows how wide a
beaker is; swapping a 250 mL for a 500 mL moves every waypoint on its own.

=============================================================================
THE GRASP IS KINEMATIC, NOT CONTACT-BASED. READ THIS BEFORE SHOWING ANYONE.
=============================================================================
When the planner commands a close and the fingertip midpoint is within
GRASP_TOLERANCE of the object's grasp keypoint, the object is rigidly attached
to the gripper and carried. On release it is handed back to the physics engine.
Contact between the fingers and the object is NOT what holds it.

This is a deliberate, labelled simplification, and it is here because
contact-based grasping does not currently work with these assets. Measured,
not assumed:

  * The YAM URDF ships nine <visual> elements and ZERO <collision> elements.
    Every arm shape is created VISIBLE-only, so by default the robot renders
    perfectly and touches nothing at all: 0 arm contacts out of 109.
  * With parse_visuals_as_colliders=True the arm gains contacts (533), but the
    colliders are now synthesized from render meshes. Each fingertip STL is an
    L-shaped part whose convex hull spans +/-36 mm, so at a 66 mm gap both
    fingers engulf a 70 mm beaker and the solver ejects it at 65 m/s.
  * Convex decomposition and force_sdf both fix the shape and neither fixes the
    grasp. Sweeping grip force over 2/3/5/10/25 N, the beaker either never
    leaves the bench or is launched; the test tube is launched every time.
    Repeat runs of an identical configuration give different answers, which is
    the real tell: this is a marginal contact problem, not a tuning problem.

Thin-walled glassware (1.0-1.5 mm) against colliders reconstructed from visual
meshes is about the worst case a penetration-based solver can be handed.

So what this demo DOES show: the generated scene and the robot share one
consistent coordinate frame, catalog keypoints are usable IK targets, the arm
reaches every one of them under its real joint limits and torque ceiling, and a
multi-step task sequence executes end to end. What it does NOT show: that a
grasp is physically feasible, or anything whatsoever about a policy.

Fixing this properly needs collision geometry for the YAM -- either from i2rt
or authored against measured pad dimensions. Set GRASP_MODE="contact" to
reproduce the failure.

    python demo_tasks.py <scene.usda> <yam.urdf> <out_dir> [--contact]
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
from arm_drive import configure_drives                                   # noqa: E402
from arm_ik import lerp_path                                             # noqa: E402
from grasp import (GraspFK, N_ARM, finger_q_for_gap, quat_to_matrix,     # noqa: E402
                   solve_grasp_ik, with_fingers)

sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG                                       # noqa: E402
from labgen.types import SceneSpec                                       # noqa: E402

WIDTH, HEIGHT = 1600, 900
FPS = 60
DT = 1.0 / 240.0
SUBSTEPS = 4

DOWN = np.array([0.0, 0.0, -1.0])
# Carry height. 0.34 was unreachable with a downward wrist -- the IK hit the
# position exactly and came back 90 degrees off on the approach axis, because
# near full extension this arm cannot keep the gripper pointing down. A rigidly
# attached object then swung with the flipped wrist and ended up under the
# worktop. 0.22 stays inside the band where a top-down pose actually exists.
TRANSIT_Z = 0.22
# How far the fingertip may be from where the plan thinks the object is.
#
# 70 mm, which is loose, and the looseness is the point. The plan plots task 3's
# grasp from where task 1 *intended* to put the beaker; the beaker actually
# settles a few millimetres off that, and by the third task the accumulated
# difference was 55 mm. Tightening this does not fix anything -- it just makes
# the chain fail earlier.
#
# The real fix is not a tolerance at all: a task sequence cannot be open-loop.
# Something has to re-observe where the object ended up before the next grasp
# is planned. That something is `identify.py` + `fit.py`, T6, which is exactly
# the stage this project has been deferring. This demo is a fairly direct
# argument for why it is not optional.
GRASP_TOLERANCE = 0.07

# Whether the ARM's own links collide with the scene.
#
# False, and that is a statement about the asset rather than a convenience. The
# YAM URDF ships no collision geometry, so the only way to give the arm
# colliders is to synthesize them from the render meshes -- and those are
# grossly oversized for this job: each fingertip hull spans +/-36 mm, wider
# than the beaker it is supposed to grasp. With them enabled the arm shoves the
# hotplate 260 mm and the test tube 455 mm just by moving past, so the objects
# are no longer where the plan left them and every grasp misses.
#
# An arm that does not collide is wrong. An arm wearing collision geometry that
# is 70 mm too wide is also wrong, and it is wrong in a way that corrupts the
# scene while looking like physics. Between two wrong options the honest one is
# the one that is obvious on inspection, so: the objects keep full physics
# against each other and the worktop, and the arm does not collide at all.
# Both of these go away once the YAM has real collision meshes.
ARM_COLLIDES = False


def aim(viewer, eye, target):
    d = np.asarray(target, float) - np.asarray(eye, float)
    viewer.set_camera(tuple(float(v) for v in eye),
                      math.degrees(math.atan2(d[2], float(np.hypot(d[0], d[1])))),
                      math.degrees(math.atan2(d[1], d[0])))


def keypoint_local(scene, instance, keypoint):
    return np.asarray(CATALOG[scene.by_id(instance).catalog_key].keypoints[keypoint])


def keypoint_world(scene, instance, keypoint):
    obj = scene.by_id(instance)
    if obj.orientation_wxyz != (1.0, 0.0, 0.0, 0.0):
        raise SystemExit(f"{instance} is rotated; not handled")
    return np.asarray(obj.position) + keypoint_local(scene, instance, keypoint)


def outer_width(instance, scene) -> float:
    d = CATALOG[scene.by_id(instance).catalog_key].dims
    return d.get("outer_d") or d.get("base_d") or max(d["x"], d["y"])


class Step:
    """One entry in the executed plan: a joint target plus what to do."""

    __slots__ = ("cfg", "attach", "release", "label")

    def __init__(self, cfg, attach=None, release=False, label=""):
        self.cfg, self.attach, self.release, self.label = cfg, attach, release, label


class Planner:
    def __init__(self, fk, lower, upper, scene):
        self.fk, self.lower, self.upper, self.scene = fk, lower, upper, scene
        self.steps: list[Step] = []
        self.q = None
        self.finger = 0.0
        self.ok = True
        # Where each object currently sits (base position). A task sequence
        # that re-grasps something it already moved has to look it up here --
        # planning against the scene file would send the arm to where the
        # object was at t=0, which is how task 3 ended up reaching 888 m for a
        # beaker that task 1 had already relocated.
        self.where = {o.instance_id: np.asarray(o.position, float)
                      for o in scene.free_bodies}

    def _ik(self, target, label, approach=DOWN):
        q, pos, ang, ok = solve_grasp_ik(self.fk, target, self.lower, self.upper,
                                         approach=approach, seed=self.q)
        print(f"   {'ok ' if ok else 'OFF'} {label:32} pos {pos * 1000:6.2f} mm  "
              f"axis {ang:5.1f} deg")
        self.q = q
        self.ok &= ok
        return q

    def _emit(self, cfgs, label="", attach=None, release=False):
        for k, c in enumerate(cfgs):
            self.steps.append(Step(c, attach if k == 0 else None,
                                   release if k == 0 else False,
                                   label if k == 0 else ""))

    def hold(self, secs, label="", attach=None, release=False):
        self._emit([with_fingers(self.q, self.finger)] * int(secs * FPS),
                   label, attach, release)

    def move_to(self, target, secs, label, approach=DOWN):
        q = self._ik(target, label, approach)
        a = with_fingers(self.steps[-1].cfg[:N_ARM] if self.steps else q, self.finger)
        self._emit(lerp_path([a, with_fingers(q, self.finger)], [secs], FPS), label)

    def set_fingers(self, gap_m, secs, label, attach=None, release=False):
        tgt = finger_q_for_gap(gap_m)
        cfgs = lerp_path([with_fingers(self.q, self.finger),
                          with_fingers(self.q, tgt)], [secs], FPS)
        self._emit(cfgs, label, attach, release)
        self.finger = tgt


PLACE_CLEARANCE = 0.004   # set the base down just above the surface, never into it


def pick_and_place(pl, obj, grasp_kp, place_xy, surface_z):
    """Grasp `obj` at `grasp_kp` and set its BASE down on `surface_z`.

    `surface_z` is the height of the thing it lands on -- the worktop at 0, the
    hotplate at its plate_center. The fingertip target is derived from it, not
    typed: surface + the keypoint's own height above the object's base + a
    clearance. Passing the fingertip height directly is how the beaker got
    placed 2 mm INSIDE the hotplate, which the solver resolved by ejecting it
    through the bench.
    """
    kp_z = float(keypoint_local(pl.scene, obj, grasp_kp)[2])
    grasp = pl.where[obj] + keypoint_local(pl.scene, obj, grasp_kp)
    width = outer_width(obj, pl.scene)
    place = np.array([place_xy[0], place_xy[1], surface_z + kp_z + PLACE_CLEARANCE])
    print(f"  {obj} ({width * 1000:.0f} mm) {grasp_kp} -> base on z={surface_z:.3f}, "
          f"fingertip ({place[0]:+.3f}, {place[1]:+.3f}, {place[2]:.3f})")
    pl.where[obj] = np.array([place_xy[0], place_xy[1], surface_z + PLACE_CLEARANCE])

    above = np.array([grasp[0], grasp[1], TRANSIT_Z])
    over = np.array([place[0], place[1], TRANSIT_Z])

    pl.set_fingers(0.089, 0.4, f"{obj}: open")
    pl.move_to(above, 1.4, f"{obj}: over object")
    pl.move_to(grasp, 1.0, f"{obj}: descend")
    # Close to the object's actual width, with no squeeze.
    #
    # A contact grasp wants a few mm of interference so the fingers load the
    # object. A kinematic hold has no contact to resolve that interference, so
    # the squeeze becomes pure geometry clipping -- the pads visibly cut into
    # the glass and the grasp looks fake even though the numbers are right.
    # Restore the squeeze if GRASP_MODE ever goes back to contact.
    pl.set_fingers(max(width, 0.006), 0.6, f"{obj}: close", attach=(obj, grasp_kp))
    pl.hold(0.25)
    pl.move_to(above, 1.0, f"{obj}: lift")
    pl.move_to(over, 1.6, f"{obj}: carry")
    pl.move_to(place, 1.1, f"{obj}: lower")
    pl.set_fingers(0.089, 0.5, f"{obj}: release", release=True)
    pl.hold(0.25)
    pl.move_to(over, 0.9, f"{obj}: withdraw")


def main() -> int:
    scene_usda, urdf, out_dir = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    contact_mode = "--contact" in sys.argv
    out_dir.mkdir(parents=True, exist_ok=True)
    scene = SceneSpec.read(
        f"/mnt/c/Ujaan Docx/Research/labgen/examples/{scene_usda.stem}.json")

    builder = newton.ModelBuilder()
    builder.add_usd(str(scene_usda))
    sc, sd = builder.joint_coord_count, builder.joint_dof_count

    arm = newton.ModelBuilder()
    arm.add_urdf(str(urdf), xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
                 floating=False, enable_self_collisions=False,
                 collapse_fixed_joints=True, up_axis=newton.Axis.Z,
                 parse_visuals_as_colliders=ARM_COLLIDES, mesh_maxhullvert=64)
    lower = np.asarray(arm.joint_limit_lower, float)
    upper = np.asarray(arm.joint_limit_upper, float)
    builder.add_builder(arm)

    dofs = slice(sd, sd + arm.joint_dof_count)
    coords = slice(sc, sc + arm.joint_coord_count)
    configure_drives(builder, dof_slice=dofs, n_fingers=2)
    print("   NOTE: arm effort limit is 40 N.m, ABOVE the YAM's rated 10 N.m. "
          "See arm_drive.py -- 10 N.m sags 48.7 deg at this reach.")

    model = builder.finalize()
    fk = GraspFK(model, coords)
    print(f"model: {model.body_count} bodies")
    print(f"grasp    : {'CONTACT (expected to fail)' if contact_mode else 'KINEMATIC -- labelled simplification'}")
    print(f"arm links: {'colliding' if ARM_COLLIDES else 'NOT colliding (no collision geometry in the URDF)'}")

    # --- plan -------------------------------------------------------------
    pl = Planner(fk, lower, upper, scene)
    # Inside the measured top-down reachable set (scripts/probe_reach.py):
    # solves 5/5 at radius 30-46 cm up to z=250 mm, and 0/5 above that at this
    # radius. The previous ready pose -- z=400 mm, radius 226 mm -- was in the
    # unreachable region, so the IK matched the position and came back with the
    # wrist 90 degrees off, and every task inherited that seed.
    ready = np.array([0.20, 0.28, 0.25])
    pl.q, _, _, _ = solve_grasp_ik(fk, ready, lower, upper, approach=DOWN)
    pl.steps.append(Step(with_fingers(pl.q, 0.0), label="ready"))
    pl.hold(0.6)

    plate = keypoint_world(scene, "hotplate", "plate_center")
    tube0 = keypoint_world(scene, "test_tube", "base_center")
    beak0 = keypoint_world(scene, "beaker", "base_center")

    print("\nTASK 1 - stand the beaker on the hotplate")
    pick_and_place(pl, "beaker", "body_grasp", (plate[0], plate[1]), plate[2] + 0.050)
    print("\nTASK 2 - move the test tube clear of the work area")
    pick_and_place(pl, "test_tube", "neck_grasp", (tube0[0] - 0.16, tube0[1] - 0.10), 0.052)
    print("\nTASK 3 - bring the beaker back to the worktop")
    pick_and_place(pl, "beaker", "body_grasp", (beak0[0] + 0.06, beak0[1] - 0.10), 0.053)

    pl.move_to(ready, 1.2, "return to ready")
    pl.hold(1.0)
    print(f"\ntrajectory: {len(pl.steps)} frames ({len(pl.steps) / FPS:.1f} s), "
          f"all IK converged = {pl.ok}")

    # --- simulate ---------------------------------------------------------
    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = pl.steps[0].cfg
    model.joint_q.assign(q_all.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)

    solver = newton.solvers.SolverMuJoCo(model, iterations=120, ls_iterations=60)
    viewer = ViewerGL(width=WIDTH, height=HEIGHT, headless=True, vsync=False)
    viewer.set_model(model)

    free = [o.instance_id for o in scene.free_bodies]
    body_of = {name: i for i, name in enumerate(free)}
    starts = {o.instance_id: np.asarray(o.position) for o in scene.free_bodies}
    intended: dict[str, np.ndarray] = {}

    targets = control.joint_target_q.numpy().copy()
    look_at = np.array([-0.02, 0.22, 0.14])
    held: tuple[int, np.ndarray, np.ndarray] | None = None
    from PIL import Image

    n = 0
    for i, step in enumerate(pl.steps):
        if step.attach and not contact_mode:
            name, kp = step.attach
            bi = body_of[name]
            bq = s0.body_q.numpy()
            gp, _ = fk.pose(step.cfg[:N_ARM])
            obj_p = bq[bi, :3]
            if np.linalg.norm(gp - obj_p - keypoint_local(scene, name, kp)) < GRASP_TOLERANCE:
                # Rigid attach: remember the object's pose in the gripper frame.
                g = bq[-3]
                R = quat_to_matrix(g[3:])
                # Carry as a fixed WORLD-frame offset from the fingertip
                # midpoint, keeping the object's own upright orientation.
                #
                # The obvious alternative -- record the object's pose in the
                # gripper frame and apply the gripper's transform -- couples the
                # carried object to wrist roll. Every IK solve picks its own
                # wrist angle, so the beaker rotated between waypoints and was
                # released 90 mm above where the plan said. Both the grasp and
                # the place use a top-down approach, so a world-frame offset is
                # exactly right here and cannot tip the glassware in mid-air.
                tip_mid = fk.grasp_point_from_state(bq)
                held = (bi, bq[bi, :3] - tip_mid, bq[bi, 3:].copy())
                print(f"   [{i}] attached {name}")
            else:
                print(f"   [{i}] attach FAILED for {name}: fingertip "
                      f"{np.linalg.norm(gp - obj_p - keypoint_local(scene, name, kp)) * 1000:.1f} mm away")
        if step.release and held is not None:
            # Hand it back to physics at rest. Without this it inherits
            # whatever velocity the solver inferred from the last carried jump.
            bi = held[0]
            jqd = s0.joint_qd.numpy()
            jqd[6 * bi: 6 * bi + 6] = 0.0
            s0.joint_qd.assign(jqd.astype(np.float32))
            z = s0.body_q.numpy()[bi, 2]
            print(f"   [{i}] released {free[bi]} at z={z * 1000:.1f} mm")
            held = None

        targets[coords] = step.cfg
        control.joint_target_q.assign(targets)
        for _ in range(SUBSTEPS):
            contacts = model.collide(s0)
            solver.step(s0, s1, control, contacts, DT)
            s0, s1 = s1, s0

        # Carry the held object ONCE per rendered frame, not once per substep.
        # Rewriting a free body's qpos four times inside one frame fights the
        # integrator: the solver keeps deriving a velocity from the jump it did
        # not make, the object accumulates speed it never physically had, and on
        # release it shoots through the worktop. (It fell 2.3 km.)
        if held is not None:
            bi, offset, quat = held
            bq_now = s0.body_q.numpy()
            pos = fk.grasp_point_from_state(bq_now) + offset
            jq = s0.joint_q.numpy()
            jq[7 * bi: 7 * bi + 3] = pos
            jq[7 * bi + 3: 7 * bi + 7] = quat
            s0.joint_q.assign(jq.astype(np.float32))
            jqd = s0.joint_qd.numpy()
            jqd[6 * bi: 6 * bi + 6] = 0.0
            s0.joint_qd.assign(jqd.astype(np.float32))

            # Write the held object's body_q directly instead of calling
            # newton.eval_fk on the whole model.
            #
            # eval_fk recomputes EVERY body from joint_q, and SolverMuJoCo does
            # not sync joint_q back for the articulation -- only body_q. So the
            # arm's joint_q is stale, and eval_fk cheerfully rebuilt the arm
            # from it, teleporting the fingertips 380 mm below the worktop. The
            # next frame then read that corrupted fingertip position to decide
            # where to carry the object, and the whole thing compounded.
            bq_w = s0.body_q.numpy()
            bq_w[bi, :3] = pos
            bq_w[bi, 3:] = quat
            s0.body_q.assign(bq_w.astype(np.float32))

        if not np.isfinite(s0.body_q.numpy()).all():
            print(f"   diverged at frame {i}")
            break

        a = -1.0 + 0.55 * math.sin(2.0 * math.pi * i / len(pl.steps))
        eye = look_at + np.array([1.05 * math.cos(a), 1.05 * math.sin(a), 0.62])
        aim(viewer, eye, look_at)
        viewer.begin_frame(i / FPS)
        viewer.log_state(s0)
        viewer.end_frame()
        f = viewer.get_frame()
        if f is None:
            break
        arr = f.numpy() if hasattr(f, "numpy") else np.asarray(f)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(arr[:, :, :3]).save(out_dir / f"frame_{i:04d}.png")
        n += 1
        if i % 300 == 0:
            jq_now = s0.joint_q.numpy()[coords]
            print(f"   frame {i}/{len(pl.steps)}  arm tracking error "
                  f"{np.degrees(np.abs(jq_now[:6] - step.cfg[:6]).max()):.2f} deg")

    viewer.close()

    print("\nwhere each object ended up:")
    final = s0.body_q.numpy()
    for name, bi in body_of.items():
        moved = np.linalg.norm(final[bi, :3] - starts[name])
        print(f"   {name:10} {np.round(starts[name], 3)} -> {np.round(final[bi, :3], 3)}"
              f"   moved {moved * 1000:6.1f} mm")
    print(f"\nwrote {n} frames")
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
