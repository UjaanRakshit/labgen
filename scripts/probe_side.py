"""Level side elevation of the grasp: camera at grasp height, looking level.

From a three-quarter view the gripper body is drawn over the beaker's upper
region and reads as intersection. Looking horizontally removes the depth
ambiguity -- vertical gaps appear as vertical gaps.
"""
import sys, math, numpy as np, newton, warp as wp
from newton.viewer import ViewerGL
from PIL import Image
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap, quat_to_matrix
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
DOWN=np.array([0,0,-1.0])
scene=SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
KPZ=CATALOG["beaker_250"].keypoints["rim_grasp"][2]
b=newton.ModelBuilder(); r=b.add_usd("bench_arm.usda")
sc,sd=b.joint_coord_count,b.joint_dof_count
a=newton.ModelBuilder()
a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
  xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
  enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
  parse_visuals_as_colliders=False, mesh_maxhullvert=64)
lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
b.add_builder(a)
coords=slice(sc,sc+a.joint_coord_count)
configure_drives(b,dof_slice=slice(sd,sd+a.joint_dof_count),n_fingers=2,verbose=False)
m=b.finalize(); fk=GraspFK(m,coords)
bpos=np.asarray(scene.by_id("beaker").position)
tgt=bpos+np.array([0,0,KPZ])
q,_,_,_=solve_grasp_ik(fk,tgt,lo,hi,approach=DOWN)
full=np.zeros(8); full[:6]=q; full[6:]=finger_q_for_gap(0.073)
s=m.state()
jq=m.joint_q.numpy().copy(); jq[coords]=full; m.joint_q.assign(jq.astype(np.float32))
newton.eval_fk(m,m.joint_q,m.joint_qd,s)
v=ViewerGL(width=1100,height=800,headless=True,vsync=False); v.set_model(m)
look=np.array([bpos[0],bpos[1],0.085])
for tag,ang in (("A",0.0),("B",math.pi/2)):
    eye=look+np.array([0.34*math.cos(ang),0.34*math.sin(ang),0.0])  # level
    d=look-eye
    v.set_camera(tuple(float(x) for x in eye),
                 math.degrees(math.atan2(d[2],float(np.hypot(d[0],d[1])))),
                 math.degrees(math.atan2(d[1],d[0])))
    v.begin_frame(0.0); v.log_state(s); v.end_frame()
    f=v.get_frame(); arr=f.numpy() if hasattr(f,'numpy') else np.asarray(f)
    if arr.dtype!=np.uint8: arr=(np.clip(arr,0,1)*255).astype(np.uint8)
    Image.fromarray(arr[:,:,:3]).save(f"side_{tag}.png")
    print("wrote side_"+tag)
v.close()
