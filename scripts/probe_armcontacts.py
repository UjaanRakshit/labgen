"""Does the arm actually collide now? Count shapes and contacts, don't eyeball it.

The YAM URDF ships zero <collision> elements, so an arm imported with
parse_visuals_as_colliders=False has no collision geometry at all and passes
through everything while rendering perfectly. That is indistinguishable from
working, in a screenshot.
"""
import sys
from pathlib import Path

import numpy as np

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasp import set_state                                    # noqa: E402
import teleop_sim as T                                                   # noqa: E402
from grasp import GraspFK, N_ARM, with_fingers                           # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS, solve_pose                          # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def count(collide: bool, target):
    model, arm, dofs, coords, lower, upper = T.build(Path(sys.argv[1]), Path(URDF),
                                                     collide=collide)
    fk = GraspFK(model, coords)
    lo, hi = lower[:N_ARM], upper[:N_ARM]
    res = solve_pose(fk, target, lo, hi, approach=T.DOWN)
    n_arm_bodies = len(np.asarray(arm.body_mass, float))
    off = model.body_count - n_arm_bodies
    shape_body = model.shape_body.numpy()
    arm_shapes = int(np.sum(shape_body >= off))

    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = with_fingers(res.q, YAM_JAWS.q_closed)
    model.joint_q.assign(q_all.astype(np.float32))
    set_state(model, s0, model.joint_q.numpy())
    set_state(model, s1, model.joint_q.numpy())
    targets = control.joint_target_q.numpy().copy()
    targets[coords] = with_fingers(res.q, YAM_JAWS.q_closed)
    control.joint_target_q.assign(targets)

    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20)
    beaker_start = None
    peak = 0
    for i in range(240):
        contacts = model.collide(s0)
        n = int(getattr(contacts, "rigid_contact_count", np.array([0])).numpy()[0]) \
            if hasattr(contacts, "rigid_contact_count") else 0
        peak = max(peak, n)
        solver.step(s0, s1, control, contacts, 1 / 120)
        s0, s1 = s1, s0
        if beaker_start is None:
            beaker_start = s0.body_q.numpy()[:off, :3].copy()
    moved = np.linalg.norm(s0.body_q.numpy()[:off, :3] - beaker_start, axis=1).max()
    return arm_shapes, peak, moved * 1000, res


def main() -> int:
    # A pose with the closed jaws driven down into where the beaker stands.
    target = np.array([0.20, 0.3464, 0.055])
    for collide in (False, True):
        shapes, peak, moved, res = count(collide, target)
        print(f"parse_visuals_as_colliders={str(collide):5}  "
              f"arm collision shapes {shapes:3d}  peak contacts {peak:5d}  "
              f"scene object moved {moved:7.2f} mm   (IK {res.position_error_m*1000:.2f} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
