"""Does SolverMuJoCo write joint_q back for the articulation?

This matters because demo_tasks.py reports an "arm tracking error" computed as
|joint_target_q - joint_q|, and the answer decides whether that number means
anything at all. This project has already been bitten once by the same gap:
newton.eval_fk rebuilt the arm from a stale joint_q and teleported the
fingertips 380 mm below the worktop.

The test is direct. Assign a home configuration, command a target far from it,
step long enough that the arm has demonstrably moved (checked against body_q,
which the solver definitely does write), and then ask whether joint_q moved too.

    joint_q unchanged  -> the "tracking error" is |target - home| and is noise
    joint_q tracks     -> the error is real and the arm genuinely lags
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from arm_drive import configure_drives                                   # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
DT = 1.0 / 240.0
STEPS = 480          # 2 s, far longer than this move needs


def main() -> int:
    b = newton.ModelBuilder()
    b.add_urdf(URDF, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
               floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=False, mesh_maxhullvert=64)
    nc, nd = b.joint_coord_count, b.joint_dof_count
    configure_drives(b, n_fingers=2, verbose=False)
    model = b.finalize()

    home = np.zeros(nc, dtype=np.float32)
    model.joint_q.assign(home)
    s0, s1 = model.state(), model.state()
    control = model.control()
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)

    # A target the arm can reach and hold, well clear of home -- and INSIDE the
    # joint limits. The first version of this probe commanded joint 2 to
    # -0.60 rad; its URDF lower limit is -8.9e-16, i.e. zero. The joint sat at
    # the limit and did not move, which looks exactly like a dead drive.
    lower = np.asarray(b.joint_limit_lower, float)
    upper = np.asarray(b.joint_limit_upper, float)
    target = home.copy()
    target[1] = 0.60           # shoulder, within [0, 3.665]
    target[2] = 0.80           # elbow, within [0, 3.142]
    for j in (1, 2):
        assert lower[j] <= target[j] <= upper[j], (
            f"joint {j + 1} target {target[j]} outside [{lower[j]}, {upper[j]}]")
    print(f"joint limits: j2 [{lower[1]:.3f}, {upper[1]:.3f}]  "
          f"j3 [{lower[2]:.3f}, {upper[2]:.3f}]")
    control.joint_target_q.assign(target)

    solver = newton.solvers.SolverMuJoCo(model, iterations=120, ls_iterations=60)

    body_start = s0.body_q.numpy().copy()
    for _ in range(STEPS):
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, DT)
        s0, s1 = s1, s0

    jq_end = s0.joint_q.numpy().copy()
    body_end = s0.body_q.numpy()
    body_moved = np.linalg.norm(body_end[:, :3] - body_start[:, :3], axis=1).max()
    jq_moved = float(np.abs(jq_end[:nc] - home[:nc]).max())

    print(f"bodies: {model.body_count}, coords {nc}, dofs {nd}")
    print(f"commanded joint 2 to {target[1]:+.3f} rad, joint 3 to {target[2]:+.3f} rad")
    print(f"body_q  moved  {body_moved * 1000:9.2f} mm   (the solver writes this)")
    print(f"joint_q moved  {np.degrees(jq_moved):9.2f} deg  (the question)")
    print(f"joint_q now:   {np.round(jq_end[:nc], 4)}")
    print(f"target was:    {np.round(target[:nc], 4)}")

    if body_moved > 0.02 and jq_moved < 1e-4:
        print("\nVERDICT: SolverMuJoCo does NOT write joint_q back. The arm moved "
              f"{body_moved * 1000:.0f} mm while joint_q stayed at its assigned value.")
        print("Any 'tracking error' computed from joint_q is |target - home| and "
              "means nothing. demo_tasks.py's arm tracking error is that number.")
        return 0
    if body_moved > 0.02:
        err = np.degrees(np.abs(jq_end[:nc] - target[:nc]).max())
        print(f"\nVERDICT: joint_q IS live. Steady-state error {err:.2f} deg, so a "
              "reported tracking error is a real lag.")
        return 0
    print("\nINCONCLUSIVE: the arm did not move, so nothing was tested.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
