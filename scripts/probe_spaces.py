import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
b = newton.ModelBuilder(); b.add_usd("bench_5.usda")
print("after bench: coord=%d dof=%d" % (b.joint_coord_count, b.joint_dof_count))
a = newton.ModelBuilder()
a.add_mjcf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
print("arm alone : coord=%d dof=%d" % (a.joint_coord_count, a.joint_dof_count))
b.add_builder(a)
print("combined  : coord=%d dof=%d" % (b.joint_coord_count, b.joint_dof_count))
m = b.finalize()
print("model joint_q len:", len(m.joint_q.numpy()), " joint_qd len:", len(m.joint_qd.numpy()))
c = m.control()
print("control.joint_target_q len:", len(c.joint_target_q.numpy()))
