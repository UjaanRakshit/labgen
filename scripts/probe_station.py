import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
S="/home/ujaan/isaac/i2rt/i2rt/robot_models/station/yam_station_linear_4310_d405/yam_station_linear_4310_d405.xml"
b = newton.ModelBuilder()
b.add_mjcf(S, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
           mesh_maxhullvert=64)
print("bodies:", [x.split('/')[-1] for x in b.body_label])
print("dof:", b.joint_dof_count, "coord:", b.joint_coord_count, "shapes:", b.shape_count)
print("limits lo:", np.round(np.array(b.joint_limit_lower),4))
print("limits hi:", np.round(np.array(b.joint_limit_upper),4))
print("joint types:", [str(t).split('.')[-1] for t in b.joint_type])
configure_drives(b, verbose=False)
m = b.finalize()
fl = m.shape_flags.numpy()
print("shape flags (unique):", sorted(set(int(x) for x in fl)), " (2=COLLIDE_SHAPES bit)")
sb = m.shape_body.numpy(); lbl=[x.split('/')[-1] for x in b.body_label]
for i in range(m.shape_count):
    bi=int(sb[i]); nm = lbl[bi] if 0<=bi<len(lbl) else f"world"
    lo=m.shape_collision_aabb_lower.numpy()[i]; hi=m.shape_collision_aabb_upper.numpy()[i]
    print(f"   shape[{i}] {nm:14} flags={int(fl[i])} size={np.round(hi-lo,4)}")
s=m.state(); newton.eval_fk(m,m.joint_q,m.joint_qd,s)
q=s.body_q.numpy()
for i,n in enumerate(lbl): print(f"   body {n:14} {np.round(q[i,:3],4)}")
