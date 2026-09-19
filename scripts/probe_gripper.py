import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
U="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
b = newton.ModelBuilder()
b.add_urdf(U, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
print("bodies:", list(b.body_label))
print("dof:", b.joint_dof_count, " coord:", b.joint_coord_count)
print("limit lower:", np.round(np.array(b.joint_limit_lower),4))
print("limit upper:", np.round(np.array(b.joint_limit_upper),4))
jt = getattr(b, "joint_type", None)
if jt is not None: print("joint types:", list(jt))
configure_drives(b, verbose=False)
m = b.finalize()
s = m.state()
for fingers in (0.0, -0.047):
    q = m.joint_q.numpy().copy(); q[6:8] = fingers
    m.joint_q.assign(q.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    p = s.body_q.numpy()
    print(f"\nfinger q={fingers}:")
    for i,n in enumerate(b.body_label):
        print(f"   {n.split('/')[-1]:14} {np.round(p[i,:3],4)}")
