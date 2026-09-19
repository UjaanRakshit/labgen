"""How much shoulder torque does holding a 0.40 m reach actually need?

joint2 saturates at the YAM's 10 N.m and sags 69 degrees. Either the arm is
genuinely too weak in this pose -- a real statement about the robot -- or
something else is eating the torque budget.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen")
import arm_drive
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, with_fingers
DOWN=np.array([0,0,-1.0])
URDF="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"

def run(effort, ke, target):
    b = newton.ModelBuilder()
    a = newton.ModelBuilder()
    a.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
    lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
    b.add_builder(a)
    configure_drives(b, ke=ke, effort=effort, n_fingers=2, verbose=False)
    m=b.finalize(); fk=GraspFK(m, slice(0,m.joint_coord_count))
    q,pos,ang,ok = solve_grasp_ik(fk, target, lo, hi, approach=DOWN)
    s0,s1=m.state(),m.state(); c=m.control()
    jq=m.joint_q.numpy().copy(); jq[:8]=with_fingers(q,0.0); m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
    tg=c.joint_target_q.numpy().copy(); tg[:8]=with_fingers(q,0.0); c.joint_target_q.assign(tg)
    solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
    for _ in range(480):
        ct=m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1=s1,s0
    err=np.abs(s0.joint_q.numpy()[:6]-q[:6])
    tip=0.5*(s0.body_q.numpy()[-1,:3]+s0.body_q.numpy()[-2,:3])
    print(f"  effort={effort:5.0f} Nm ke={ke:6.0f}: max joint err={err.max():.4f} rad "
          f"({np.degrees(err.max()):5.1f} deg)  worst j{int(err.argmax())+1}  "
          f"tip err={np.linalg.norm(tip-target)*1000:6.1f} mm")

print("total arm mass:", end=" ")
_b = newton.ModelBuilder()
_b.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
            enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
print(f"{sum(_b.body_mass):.3f} kg  per-link {np.round(np.array(_b.body_mass),3)}")
T = np.array([0.20, 0.346, 0.052])   # the beaker grasp
print(f"holding the beaker grasp pose {T}:")
for eff in (10, 20, 40, 100):
    run(eff, 300.0, T)
print("with a stiffer arm gain:")
for ke in (1000.0, 3000.0):
    run(40, ke, T)
