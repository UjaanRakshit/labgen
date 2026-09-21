"""How fast can the IK actually run? Teleop is useless if the answer is 4 Hz.

The Jacobian is central differences over 6 joints, so every iteration costs 12
forward-kinematics evaluations, and each of those is an eval_fk over the whole
model. That is fine for planning a trajectory offline and possibly fatal at
60 Hz, so it gets measured before a control loop is designed around it.

Warm starting is the thing being tested. During teleop the target moves a
millimetre or two per tick, so the previous configuration is an excellent seed
and the solver should need very few iterations -- unlike planning, where it
starts from scratch with multiple restarts.
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasp import GraspFK, N_ARM                                         # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import solve_pose                                    # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def main() -> int:
    b = newton.ModelBuilder()
    b.add_urdf(URDF, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
               floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=False, mesh_maxhullvert=64)
    lo = np.asarray(b.joint_limit_lower, float)[:N_ARM]
    hi = np.asarray(b.joint_limit_upper, float)[:N_ARM]
    model = b.finalize()
    fk = GraspFK(model, slice(0, model.joint_coord_count))

    q = 0.5 * (lo + hi)
    base, _ = fk.pose(q)

    n = 200
    t0 = time.perf_counter()
    for _ in range(n):
        fk.pose(q)
    fk_ms = (time.perf_counter() - t0) / n * 1000
    print(f"one forward-kinematics evaluation: {fk_ms:.3f} ms")
    print(f"one Jacobian (12 evals):           {fk_ms * 12:.2f} ms")
    print()

    print(f"{'iters':>6} {'restarts':>9} {'ms/solve':>9} {'Hz':>7} "
          f"{'err (mm)':>9}  warm-started 2 mm step")
    for iters in (1, 2, 3, 5, 10, 20):
        qc = q.copy()
        errs, t0 = [], time.perf_counter()
        reps = 40
        for i in range(reps):
            target = base + np.array([0.002 * ((i % 10) - 5), 0.0, 0.0])
            res = solve_pose(fk, target, lo, hi, seed=qc, restarts=1,
                             iters=iters, tol_m=0.001)
            qc = res.q
            errs.append(res.position_error_m * 1000)
        ms = (time.perf_counter() - t0) / reps * 1000
        print(f"{iters:6d} {1:9d} {ms:9.2f} {1000/ms:7.1f} {np.mean(errs):9.3f}")

    print()
    print("cold start, as the planner uses it (10 restarts, 160 iters):")
    t0 = time.perf_counter()
    res = solve_pose(fk, base + np.array([0.05, 0.0, 0.0]), lo, hi)
    print(f"   {(time.perf_counter()-t0)*1000:.0f} ms, err "
          f"{res.position_error_m*1000:.3f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
