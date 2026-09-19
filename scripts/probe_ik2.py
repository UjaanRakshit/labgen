import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, N_ARM
U="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
b = newton.ModelBuilder()
b.add_urdf(U, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
configure_drives(b, verbose=False)
m = b.finalize()
lo = np.asarray(b.joint_limit_lower,float); hi = np.asarray(b.joint_limit_upper,float)
fk = GraspFK(m, slice(0, m.joint_coord_count))
DOWN = np.array([0,0,-1.0])
tgt = np.array([0.2, 0.3464, 0.0522])   # beaker body_grasp in bench_arm
print("target:", tgt, " |r| =", round(float(np.linalg.norm(tgt)),3))
for label, ap, w in (("position only", None, 0.05),
                     ("down w=0.05", DOWN, 0.05),
                     ("down w=0.20", DOWN, 0.20),
                     ("down w=0.50", DOWN, 0.50)):
    q,pos,ang,ok = solve_grasp_ik(fk, tgt, lo, hi, approach=ap, w_axis=w, restarts=12)
    print(f"  {label:16} pos={pos*1000:7.2f} mm  axis={ang:6.1f} deg  ok={ok}")
