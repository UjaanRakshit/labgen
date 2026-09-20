"""Penetration measured on the ACTUAL simulated carry, not a fresh IK solve.

The earlier probe solved IK afresh and reported 0 vertices inside the beaker.
The render disagrees, so the probe was measuring a different configuration from
the one that runs. Replay task 1 and check the real state, per body, so it is
clear WHICH part of the hand is in the glass.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from arm_ik import lerp_path
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap, quat_to_matrix
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
DOWN=np.array([0,0,-1.0]); FPS=60; DT=1/240; SUB=4
KP="rim_grasp"
scene=SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
R=CATALOG["beaker_250"].dims["outer_d"]/2; H=CATALOG["beaker_250"].dims["height"]
b=newton.ModelBuilder(); r=b.add_usd("bench_arm.usda")
sc,sd=b.joint_coord_count,b.joint_dof_count
a=newton.ModelBuilder()
a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
  xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
  enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
  parse_visuals_as_colliders=False, mesh_maxhullvert=64)
lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
b.add_builder(a)
coords=slice(sc,sc+a.joint_coord_count); dofs=slice(sd,sd+a.joint_dof_count)
configure_drives(b,dof_slice=dofs,n_fingers=2,verbose=False)
m=b.finalize(); fk=GraspFK(m,coords)
paths={v:k for k,v in r["path_body_map"].items()}
bk=[i for i in range(4) if paths[i].endswith("beaker")][0]
lbl=[x.split('/')[-1] for x in a.body_label]
sb=m.shape_body.numpy(); sxf=m.shape_transform.numpy()
NSCENE=4

def per_body(bodies, centre):
    rows=[]
    for i in range(m.shape_count):
        bi=int(sb[i])
        if bi < NSCENE: continue
        g=m.shape_source[i]
        if g is None or not hasattr(g,'vertices'): continue
        v=np.asarray(g.vertices,float)[::3]
        xf=sxf[i]
        v=v@quat_to_matrix(xf[3:]).T+xf[:3]
        v=v@quat_to_matrix(bodies[bi,3:]).T+bodies[bi,:3]
        rad=np.linalg.norm(v[:,:2]-centre[:2],axis=1)
        ins=(rad<R)&(v[:,2]>centre[2])&(v[:,2]<centre[2]+H)
        if ins.any():
            rows.append((lbl[bi-NSCENE], int(ins.sum()), (R-rad[ins]).max()*1000))
    return rows

g=np.asarray(scene.by_id("beaker").position)+np.asarray(CATALOG["beaker_250"].keypoints[KP])
qa,_,_,_=solve_grasp_ik(fk,np.array([g[0],g[1],0.22]),lo,hi,approach=DOWN)
qg,_,_,_=solve_grasp_ik(fk,g,lo,hi,approach=DOWN,seed=qa)
gq=finger_q_for_gap(0.073)  # width + 3 mm clearance
path=([with_fingers(qa,finger_q_for_gap(0.094))]*20
     +lerp_path([with_fingers(qa,finger_q_for_gap(0.094)),with_fingers(qg,finger_q_for_gap(0.094))],[1.2],FPS)
     +lerp_path([with_fingers(qg,finger_q_for_gap(0.094)),with_fingers(qg,gq)],[0.6],FPS)
     +[with_fingers(qg,gq)]*15
     +lerp_path([with_fingers(qg,gq),with_fingers(qa,gq)],[1.2],FPS)+[with_fingers(qa,gq)]*30)
s0,s1=m.state(),m.state(); c=m.control()
q0=m.joint_q.numpy().copy(); q0[coords]=path[0]; m.joint_q.assign(q0.astype(np.float32))
newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
tg=c.joint_target_q.numpy().copy(); held=None; ATT=int(20+1.2*FPS+0.6*FPS)
for i,cfg in enumerate(path):
    if i==ATT:
        bq=s0.body_q.numpy(); held=(bk,bq[bk,:3]-fk.grasp_point_from_state(bq),bq[bk,3:].copy())
    tg[coords]=cfg; c.joint_target_q.assign(tg)
    for _ in range(SUB):
        ct=m.collide(s0); solver.step(s0,s1,c,ct,DT); s0,s1=s1,s0
    if held is not None:
        bq=s0.body_q.numpy(); pos=fk.grasp_point_from_state(bq)+held[1]
        jq=s0.joint_q.numpy(); jq[7*bk:7*bk+3]=pos; jq[7*bk+3:7*bk+7]=held[2]
        s0.joint_q.assign(jq.astype(np.float32))
        jd=s0.joint_qd.numpy(); jd[6*bk:6*bk+6]=0.0; s0.joint_qd.assign(jd.astype(np.float32))
        bw=s0.body_q.numpy(); bw[bk,:3]=pos; bw[bk,3:]=held[2]; s0.body_q.assign(bw.astype(np.float32))
    if i in (ATT-1, ATT+20, len(path)-1):
        bq=s0.body_q.numpy()
        rows=per_body(bq, bq[bk,:3])
        tag = "just before close" if i==ATT-1 else ("holding" if i==ATT+20 else "lifted")
        print(f"  {tag:18} beaker_base_z={bq[bk,2]*1000:7.1f} mm")
        if not rows: print("      no hand geometry inside the beaker")
        for nm,n,d in rows: print(f"      {nm:11} {n:5d} verts inside, up to {d:6.2f} mm deep")
