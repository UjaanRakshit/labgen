"""Where does the 277 mm startup transient come from?

With the real robot's soft gains the teleop sim showed a 277.7 mm peak between
where the hand was told to be and where it physically was, all inside the first
two seconds, then settled to a few millimetres. The IK residual reported 0.00 mm
throughout, so the transient is physical: the arm moved somewhere it was not
commanded to go.

This holds the ready pose with no input at all -- the same build, drives and
feedforward as teleop_sim -- and prints the physical hand deviation every 50 ms.
If it swings with no input, the transient is in how the sim starts, not in
teleop.
"""
import sys
from pathlib import Path

import numpy as np

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
import teleop_sim as T                                                    # noqa: E402
from arm_drive import gravity_torques                                     # noqa: E402
from grasp import set_state, GraspFK, N_ARM, with_fingers                            # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS, solve_pose                          # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def main() -> int:
    import warp as wp
    p = newton.ModelBuilder()
    p.add_urdf(URDF, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
               floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=False)
    lo_all = np.asarray(p.joint_limit_lower, float)
    hi_all = np.asarray(p.joint_limit_upper, float)
    rig = T.default_rig(False, lo_all, hi_all)

    for collide in (False, True):
        model, arms = T.build(Path(sys.argv[1]), Path(URDF), rig, collide=collide)
        inst = arms[0]
        inst.fk = GraspFK(model, inst.coords, body_offset=inst.body_offset,
                          n_bodies=len(inst.masses))
        q = solve_pose(inst.fk, inst.ready, inst.lo, inst.hi, approach=T.DOWN).q
        want, _ = inst.fk.pose(q)

        s0, s1 = model.state(), model.state()
        control = model.control()
        qa = model.joint_q.numpy().copy()
        qa[inst.coords] = with_fingers(q, YAM_JAWS.q_open)
        set_state(model, s0, qa)
        set_state(model, s1, qa)
        tq = control.joint_target_q.numpy().copy()
        tq[inst.coords] = with_fingers(q, YAM_JAWS.q_open)
        control.joint_target_q.assign(tq)
        ff = control.joint_f.numpy().copy()
        ff[inst.dofs.start: inst.dofs.start + N_ARM] = -gravity_torques(
            inst.fk._eval(q), inst.masses, inst.axes, coms=inst.coms,
            offset=inst.body_offset, n_arm=N_ARM)
        control.joint_f.assign(ff.astype(np.float32))

        solver = newton.solvers.SolverMuJoCo(model, iterations=10, ls_iterations=20,
                                             nconmax=T.NCONMAX, njmax=T.NJMAX)
        print(f"collide={collide}: physical hand deviation from the ready pose, "
              f"no input (mm)")
        row = []
        peak = 0.0
        for k in range(int(3.0 / T.DT)):
            contacts = model.collide(s0)
            solver.step(s0, s1, control, contacts, T.DT)
            s0, s1 = s1, s0
            d = float(np.linalg.norm(inst.fk.grasp_point_from_state(s0.body_q.numpy()) - want)) * 1000
            peak = max(peak, d)
            if k % 6 == 0:
                row.append(f"{k * T.DT:4.2f}s:{d:6.1f}")
        for i in range(0, len(row), 6):
            print("   " + "  ".join(row[i:i + 6]))
        print(f"   peak {peak:.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
