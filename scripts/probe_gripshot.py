"""Render just the hand, large, at open and closed. Look at it."""
import sys, math, numpy as np, newton, warp as wp
from newton.viewer import ViewerGL
from PIL import Image
sys.path.insert(0,"/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, quat_to_matrix, N_ARM
b=newton.ModelBuilder()
b.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
  xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
  enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
  parse_visuals_as_colliders=True, mesh_maxhullvert=64)
configure_drives(b, n_fingers=2, verbose=False)
m=b.finalize(); fk=GraspFK(m, slice(0,m.joint_coord_count))
v=ViewerGL(width=1000,height=750,headless=True,vsync=False); v.set_model(m)
for tag,q in (("open",0.0),("mid",-0.024),("closed",-0.047)):
    jq=m.joint_q.numpy().copy(); jq[6:8]=q; m.joint_q.assign(jq.astype(np.float32))
    s=m.state(); newton.eval_fk(m,m.joint_q,m.joint_qd,s)
    bodies=s.body_q.numpy(); gp=fk.grasp_point_from_state(bodies)
    L,R=m.body_count-2,m.body_count-1
    lat=float(np.linalg.norm((bodies[L,:3]-bodies[R,:3])[:2]))
    print(f"{tag:7} q={q:+.4f} grasp_pt={np.round(gp,4)} "
          f"tipL={np.round(bodies[L,:3],4)} tipR={np.round(bodies[R,:3],4)} "
          f"lateral_origin_sep={lat*1000:.1f} mm")
    for view,off in (("side",(0.0,-0.22,0.02)),("top",(0.0,-0.02,0.22))):
        eye=gp+np.array(off); d=gp-eye
        v.set_camera(tuple(float(x) for x in eye),
                     math.degrees(math.atan2(d[2],float(np.hypot(d[0],d[1])))),
                     math.degrees(math.atan2(d[1],d[0])))
        v.begin_frame(0.0); v.log_state(s); v.end_frame()
        f=v.get_frame(); arr=f.numpy() if hasattr(f,'numpy') else np.asarray(f)
        if arr.dtype!=np.uint8: arr=(np.clip(arr,0,1)*255).astype(np.uint8)
        Image.fromarray(arr[:,:,:3]).save(f"grip_{tag}_{view}.png")
v.close()
