import sys, numpy as np, newton, warp as wp
D="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"
b = newton.ModelBuilder()
b.add_mjcf(D, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
print("body keys:", [a for a in dir(b) if "body_key" in a or "body_name" in a or a=="body_label"])
for attr in ("body_key","body_name","body_label"):
    v = getattr(b, attr, None)
    if v: print(f"{attr}: {list(v)}")
print("joint limits lower:", np.round(np.array(b.joint_limit_lower),3))
print("joint limits upper:", np.round(np.array(b.joint_limit_upper),3))
m = b.finalize()
s = m.state(); newton.eval_fk(m, m.joint_q, m.joint_qd, s)
q = s.body_q.numpy()
for i in range(m.body_count):
    print(f"  body[{i}] pos={np.round(q[i,:3],4)}")
