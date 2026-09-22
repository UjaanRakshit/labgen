"""Does the arm hold the teleop ready pose, and is the feedforward sign right?

The teleop viewer showed the arm slumped on the bench while the IK reported a
perfect 0.00 mm solve for the ready pose. Either the target never reached the
drive, or the drive could not hold it. A wrong-signed gravity feedforward would
DOUBLE the gravity load rather than cancel it, and would look exactly like this.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                   # noqa: E402
from arm_drive import gravity_torques                                    # noqa: E402
from grasp import GraspFK, N_ARM, with_fingers                           # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS, solve_pose                          # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def run(dt, substeps, ff_sign, seconds=2.0):
    model, arm, dofs, coords, lower, upper = T.build(Path(sys.argv[1]), Path(URDF))
    fk = GraspFK(model, coords)
    lo, hi = lower[:N_ARM], upper[:N_ARM]
    q_cmd = solve_pose(fk, T.READY_POSE, lo, hi, approach=T.DOWN).q

    s0, s1 = model.state(), model.state()
    control = model.control()
    q_all = model.joint_q.numpy().copy()
    q_all[coords] = with_fingers(q_cmd, YAM_JAWS.q_open)
    model.joint_q.assign(q_all.astype(np.float32))
    newton.eval_fk(model, model.joint_q, model.joint_qd, s0)
    newton.eval_fk(model, model.joint_q, model.joint_qd, s1)

    targets = control.joint_target_q.numpy().copy()
    targets[coords] = with_fingers(q_cmd, YAM_JAWS.q_open)
    control.joint_target_q.assign(targets)

    masses = np.asarray(arm.body_mass, float)
    axes = np.asarray(arm.joint_axis, float)
    coms = np.asarray(arm.body_com, float)
    off = model.body_count - len(masses)
    ff = control.joint_f.numpy().copy()
    if ff_sign != 0:
        bw = fk._eval(q_cmd)
        ff[dofs.start: dofs.start + N_ARM] = ff_sign * gravity_torques(
            bw, masses, axes, coms=coms, offset=off, n_arm=N_ARM)
        control.joint_f.assign(ff.astype(np.float32))

    solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20)
    for _ in range(int(seconds / dt)):
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0

    jq = s0.joint_q.numpy()[coords][:N_ARM]
    return float(np.degrees(np.abs(jq - q_cmd[:N_ARM]).max()))


def main() -> int:
    print(f"{'dt':>10} {'substeps':>9} {'feedforward':>14}  worst joint error (deg)")
    for dt, sub in ((1 / 120, 2), (1 / 240, 4)):
        for sign, label in ((0, "none"), (-1.0, "-gravity"), (+1.0, "+gravity")):
            err = run(dt, sub, sign)
            print(f"    1/{1/dt:5.0f} {sub:9d} {label:>14}  {err:8.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
