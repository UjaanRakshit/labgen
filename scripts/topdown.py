import sys, math, numpy as np, newton, warp as wp
from newton.viewer import ViewerGL
from PIL import Image
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
b = newton.ModelBuilder(); b.add_usd("bench_arm.usda")
sc, sd = b.joint_coord_count, b.joint_dof_count
a = newton.ModelBuilder()
a.add_mjcf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
b.add_builder(a); configure_drives(b, dof_slice=slice(sd,sd+a.joint_dof_count), verbose=False)
m = b.finalize(); s = m.state(); newton.eval_fk(m, m.joint_q, m.joint_qd, s)
v = ViewerGL(width=1200, height=800, headless=True, vsync=False); v.set_model(m)
for tag, eye, look in (("t0_high", (0.05,-0.35,1.05), (0.0,0.33,0.03)),
                       ("t0_low",  (0.05,-0.55,0.22), (0.0,0.33,0.06))):
    d = np.array(look)-np.array(eye)
    v.set_camera(tuple(float(x) for x in eye),
                 math.degrees(math.atan2(d[2], float(np.hypot(d[0],d[1])))),
                 math.degrees(math.atan2(d[1], d[0])))
    v.begin_frame(0.0); v.log_state(s); v.end_frame()
    f = v.get_frame(); arr = f.numpy() if hasattr(f,"numpy") else np.asarray(f)
    if arr.dtype != np.uint8: arr = (np.clip(arr,0,1)*255).astype(np.uint8)
    Image.fromarray(arr[:,:,:3]).save(f"{tag}.png")
    print("wrote", tag)
v.close()
