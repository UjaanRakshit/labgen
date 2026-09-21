"""How deep does the gripper actually go into the beaker? Millimetres, not vibes.

A finger drawn in front of a cylinder looks identical to a finger inside it.
This measures the real 3D distance from every finger collision vertex to the
beaker surface, so "collisions do not work" becomes a number with a sign.

Negative = penetrating. Positive = clear.
"""
import sys
from pathlib import Path

import numpy as np

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                    # noqa: E402
from grasp import GraspFK, N_ARM, quat_to_matrix, with_fingers            # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG                                        # noqa: E402
from labgen.control import YAM_JAWS, solve_pose                           # noqa: E402
from labgen.types import SceneSpec                                        # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def finger_world_points(model, state, n_arm_bodies):
    """Every collision vertex of the two fingertip bodies, in world."""
    shape_body = model.shape_body.numpy()
    shape_xf = model.shape_transform.numpy()
    bq = state.body_q.numpy()
    tips = (model.body_count - 1, model.body_count - 2)
    pts = []
    for i in range(model.shape_count):
        bi = int(shape_body[i])
        if bi not in tips:
            continue
        geo = model.shape_source[i]
        if geo is None or not hasattr(geo, "vertices"):
            continue
        v = np.asarray(geo.vertices, float)
        xf = shape_xf[i]
        v = v @ quat_to_matrix(xf[3:]).T + xf[:3]
        v = v @ quat_to_matrix(bq[bi, 3:]).T + bq[bi, :3]
        pts.append(v)
    return np.vstack(pts) if pts else np.zeros((0, 3))


def signed_clearance(points, centre_xy, outer_r, top_z, base_z):
    """Signed distance to a solid cylinder wall+floor. Negative inside."""
    radial = np.linalg.norm(points[:, :2] - centre_xy, axis=1) - outer_r
    below_top = points[:, 2] - top_z          # >0 above the rim: always clear
    outside = np.maximum(radial, below_top)
    outside = np.where(points[:, 2] < base_z, np.abs(points[:, 2] - base_z), outside)
    return outside


def main() -> int:
    scene = SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
    obj = scene.by_id("beaker")
    item = CATALOG[obj.catalog_key]
    centre = np.asarray(obj.position, float)[:2]
    outer_r = item.dims["outer_d"] / 2.0
    top_z = obj.position[2] + item.dims["height"]
    print(f"beaker: centre {centre}, outer radius {outer_r*1000:.1f} mm, "
          f"rim {top_z*1000:.1f} mm")

    for collide in (False, True):
        model, arm, dofs, coords, lower, upper = T.build(
            Path(sys.argv[1]), Path(URDF), collide=collide)
        fk = GraspFK(model, coords)
        lo, hi = lower[:N_ARM], upper[:N_ARM]
        n_arm_bodies = len(np.asarray(arm.body_mass, float))

        # Drive the open jaws straight down over the beaker, the motion that
        # produced the screenshot.
        target = np.array([obj.position[0], obj.position[1], 0.088])
        res = solve_pose(fk, target, lo, hi, approach=T.DOWN)

        s0, s1 = model.state(), model.state()
        control = model.control()
        q_all = model.joint_q.numpy().copy()
        q_all[coords] = with_fingers(res.q, YAM_JAWS.q_open)
        model.joint_q.assign(q_all.astype(np.float32))
        newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
        newton.eval_fk(model, model.joint_q, model.joint_qd, s1)
        targets = control.joint_target_q.numpy().copy()
        targets[coords] = with_fingers(res.q, YAM_JAWS.q_open)
        control.joint_target_q.assign(targets)

        solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20)
        for _ in range(180):
            contacts = model.collide(s0)
            solver.step(s0, s1, control, contacts, 1 / 120)
            s0, s1 = s1, s0

        pts = finger_world_points(model, s0, n_arm_bodies)
        clear = signed_clearance(pts, centre, outer_r, top_z, obj.position[2])
        worst = float(clear.min()) * 1000
        inside = int((clear < 0).sum())
        beaker_i = [i for i, o in enumerate(scene.free_bodies)
                    if o.instance_id == "beaker"][0]
        moved = float(np.linalg.norm(
            s0.body_q.numpy()[beaker_i, :3] - np.asarray(obj.position, float))) * 1000
        print(f"colliders={str(collide):5}  finger vertices {len(pts):5d}  "
              f"inside the glass {inside:5d}  worst {worst:8.2f} mm  "
              f"beaker moved {moved:7.2f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
