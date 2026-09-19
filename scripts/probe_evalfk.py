import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
sc, sd = b.joint_coord_count, b.joint_dof_count
a = newton.ModelBuilder()
a.add_mjcf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
b.add_builder(a)
configure_drives(b, dof_slice=slice(sd, sd+a.joint_dof_count), verbose=False)
m = b.finalize()
paths = {v:k for k,v in r["path_body_map"].items()}
s = m.state()
print("bench joint_q (first 28 coords, 7 per free body):")
print(" ", np.round(m.joint_q.numpy()[:28], 4))
print("\nbody_q BEFORE eval_fk:")
for i in range(m.body_count):
    print(f"  [{i}] {paths.get(i,'arm')[:28]:30} {np.round(s.body_q.numpy()[i],4)}")
newton.eval_fk(m, m.joint_q, m.joint_qd, s)
print("\nbody_q AFTER eval_fk:")
for i in range(m.body_count):
    print(f"  [{i}] {paths.get(i,'arm')[:28]:30} {np.round(s.body_q.numpy()[i],4)}")
