"""During carry, is the object actually where the fingers are?

Prints the distance between the grasp point (pad centroid) and the object's
grasp KEYPOINT in world. If the carry is right these coincide to within the
arm's tracking error; anything larger is the object floating relative to the
hand, which is what "0 sync" looks like on screen.
"""
import sys, math, numpy as np, newton, warp as wp
from newton.viewer import ViewerGL
from PIL import Image
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from arm_ik import lerp_path
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap, N_ARM
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
DOWN=np.array([0,0,-1.0]); FPS=60; DT=1/240; SUB=4
scene=SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
b=newton.ModelBuilder(); r=b.add_usd("bench_arm.usda")
sc,sd=b.joint_coord_count,b.joint_dof_count
a=newton.ModelBuilder()
a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
  xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
  enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
  parse_visuals_as_colliders=False, mesh_maxhullvert=64)  # match the demo
lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
b.add_builder(a)
coords=slice(sc,sc+a.joint_coord_count); dofs=slice(sd,sd+a.joint_dof_count)
configure_drives(b,dof_slice=dofs,n_fingers=2,verbose=False)
m=b.finalize(); fk=GraspFK(m,coords)
paths={v:k for k,v in r["path_body_map"].items()}
bk=[i for i in range(4) if paths[i].endswith("beaker")][0]
kp=np.asarray(CATALOG["beaker_250"].keypoints["rim_grasp"])
g=np.asarray(scene.by_id("beaker").position)+kp
qa,_,_,_=solve_grasp_ik(fk,g+[0,0,0.16],lo,hi,approach=DOWN)
qg,_,_,_=solve_grasp_ik(fk,g,lo,hi,approach=DOWN,seed=qa)
gq=finger_q_for_gap(0.070)
path=([with_fingers(qa,0.0)]*20+lerp_path([with_fingers(qa,0.0),with_fingers(qg,0.0)],[1.2],FPS)
     +lerp_path([with_fingers(qg,0.0),with_fingers(qg,gq)],[0.6],FPS)+[with_fingers(qg,gq)]*15
     +lerp_path([with_fingers(qg,gq),with_fingers(qa,gq)],[1.2],FPS)+[with_fingers(qa,gq)]*40)
s0,s1=m.state(),m.state(); c=m.control()
q=m.joint_q.numpy().copy(); q[coords]=with_fingers(qa,0.0); m.joint_q.assign(q.astype(np.float32))
newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
v=ViewerGL(width=1100,height=750,headless=True,vsync=False); v.set_model(m)
tg=c.joint_target_q.numpy().copy(); held=None; ATT=int(20+1.2*FPS+0.6*FPS)
for i,cfg in enumerate(path):
    if i==ATT:
        bq=s0.body_q.numpy(); held=(bk, bq[bk,:3]-fk.grasp_point_from_state(bq), bq[bk,3:].copy())
        print(f"attach: offset={np.round(held[1],4)}  (expect ~[0,0,-0.052])")
    tg[coords]=cfg; c.joint_target_q.assign(tg)
    for _ in range(SUB):
        ct=m.collide(s0); solver.step(s0,s1,c,ct,DT); s0,s1=s1,s0
    if held is not None:
        bq=s0.body_q.numpy(); pos=fk.grasp_point_from_state(bq)+held[1]
        jq=s0.joint_q.numpy(); jq[7*bk:7*bk+3]=pos; jq[7*bk+3:7*bk+7]=held[2]
        s0.joint_q.assign(jq.astype(np.float32))
        jd=s0.joint_qd.numpy(); jd[6*bk:6*bk+6]=0.0; s0.joint_qd.assign(jd.astype(np.float32))
        bw=s0.body_q.numpy(); bw[bk,:3]=pos; bw[bk,3:]=held[2]; s0.body_q.assign(bw.astype(np.float32))
    if i in (ATT+40, ATT+90, len(path)-1):
        bq=s0.body_q.numpy(); gp=fk.grasp_point_from_state(bq)
        kpw=bq[bk,:3]+kp
        print(f"  f{i}: grasp_pt={np.round(gp,4)} obj_keypoint={np.round(kpw,4)} "
              f"-> {np.linalg.norm(gp-kpw)*1000:5.1f} mm apart")
        # close-up
        look=gp; eye=look+np.array([0.16,-0.26,0.10])
        d=look-eye
        v.set_camera(tuple(float(x) for x in eye),
                     math.degrees(math.atan2(d[2],float(np.hypot(d[0],d[1])))),
                     math.degrees(math.atan2(d[1],d[0])))
        v.begin_frame(i/FPS); v.log_state(s0); v.end_frame()
        fr=v.get_frame(); arr=fr.numpy() if hasattr(fr,'numpy') else np.asarray(fr)
        if arr.dtype!=np.uint8: arr=(np.clip(arr,0,1)*255).astype(np.uint8)
        Image.fromarray(arr[:,:,:3]).save(f"closeup_{i}.png")
v.close()
