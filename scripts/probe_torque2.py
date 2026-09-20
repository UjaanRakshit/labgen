"""Re-measure the holding torque with everything else now correct.

The original measurement was taken when GraspFK still used the tip BODY
ORIGINS, so the IK was solving for a pose 36 mm off, and the drive gains were
ke=300. Static gravity at the real grasp pose is 7.26 N.m, inside the rated 10,
so the arm should hold. If it does, "needs 20 N.m" was never a fact about the
arm.
"""
import sys, math, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
import arm_drive
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, with_fingers, N_ARM
DOWN = np.array([0, 0, -1.0])
URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
TARGET = np.array([0.20, 0.3464, 0.083])     # beaker rim_grasp, 0.409 m reach

def run(effort, ke, kd):
    b = newton.ModelBuilder()
    b.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
    lo = np.asarray(b.joint_limit_lower, float); hi = np.asarray(b.joint_limit_upper, float)
    configure_drives(b, ke=ke, kd=kd, effort=effort, n_fingers=2, verbose=False)
    m = b.finalize(); fk = GraspFK(m, slice(0, m.joint_coord_count))
    q, pos, ang, ok = solve_grasp_ik(fk, TARGET, lo, hi, approach=DOWN)
    s0, s1 = m.state(), m.state(); c = m.control()
    jq = m.joint_q.numpy().copy(); jq[:8] = with_fingers(q, 0.0)
    m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0); newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
    tg = c.joint_target_q.numpy().copy(); tg[:8] = with_fingers(q, 0.0)
    c.joint_target_q.assign(tg)
    solver = newton.solvers.SolverMuJoCo(m, iterations=120, ls_iterations=60)
    for _ in range(int(2.0 * 960)):
        ct = m.collide(s0); solver.step(s0, s1, c, ct, 1/960); s0, s1 = s1, s0
    err = np.abs(s0.joint_q.numpy()[:N_ARM] - q[:N_ARM])
    tip = fk.grasp_point_from_state(s0.body_q.numpy())
    return math.degrees(err.max()), np.linalg.norm(tip - TARGET) * 1000

print("static gravity torque at this pose: 7.26 N.m (shoulder), rated limit 10 N.m")
print(f"\n{'effort':>8} {'ke':>7} {'kd':>6} {'max sag':>10} {'tip error':>11}")
for effort, ke, kd in ((10, 300, 30), (10, 3000, 150), (10, 8000, 300),
                       (40, 3000, 150)):
    sag, tip = run(effort, ke, kd)
    verdict = "holds" if sag < 2.0 else "SAGS"
    print(f"{effort:8.0f} {ke:7.0f} {kd:6.0f} {sag:9.2f}d {tip:10.1f}mm  {verdict}")
