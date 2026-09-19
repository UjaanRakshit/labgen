import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
import arm_drive
from arm_drive import configure_drives
from arm_ik import lerp_path
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
DOWN=np.array([0,0,-1.0]); FPS=60; DT=1/240; SUB=4
scene = SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
URDF="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"

def trial(force, squeeze_mm, trace=False, obj="beaker", kp="body_grasp", width=0.070):
    arm_drive.FINGER_EFFORT_N = force
    b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
    sc, sd = b.joint_coord_count, b.joint_dof_count
    a = newton.ModelBuilder()
    a.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=True, mesh_maxhullvert=64)
    a.approximate_meshes(method="coacd")
    lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
    b.add_builder(a)
    coords=slice(sc,sc+a.joint_coord_count); dofs=slice(sd,sd+a.joint_dof_count)
    configure_drives(b, dof_slice=dofs, n_fingers=2, verbose=False)
    m=b.finalize()
    paths={v:k for k,v in r["path_body_map"].items()}
    bk=[i for i in range(4) if paths[i].endswith(obj)][0]
    fk=GraspFK(m,coords)
    g=np.asarray(scene.by_id(obj).position)+np.asarray(CATALOG[scene.by_id(obj).catalog_key].keypoints[kp])
    qa,_,_,_=solve_grasp_ik(fk,g+[0,0,0.14],lo,hi,approach=DOWN)
    qg,_,_,_=solve_grasp_ik(fk,g,lo,hi,approach=DOWN,seed=qa)
    gq=finger_q_for_gap(width-squeeze_mm/1000.0)
    path=([with_fingers(qa,0.0)]*20
          + lerp_path([with_fingers(qa,0.0),with_fingers(qg,0.0)],[1.0],FPS)
          + lerp_path([with_fingers(qg,0.0),with_fingers(qg,gq)],[0.8],FPS)
          + [with_fingers(qg,gq)]*20
          + lerp_path([with_fingers(qg,gq),with_fingers(qa,gq)],[1.2],FPS)
          + [with_fingers(qa,gq)]*60)
    s0,s1=m.state(),m.state(); c=m.control()
    q=m.joint_q.numpy().copy(); q[coords]=with_fingers(qa,0.0); m.joint_q.assign(q.astype(np.float32))
    newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
    solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
    tg=c.joint_target_q.numpy().copy(); zs=[]
    for i,cfg in enumerate(path):
        tg[coords]=cfg; c.joint_target_q.assign(tg)
        for _ in range(SUB):
            ct=m.collide(s0); solver.step(s0,s1,c,ct,DT); s0,s1=s1,s0
        z=s0.body_q.numpy()[bk,2]; zs.append(z)
        if not np.isfinite(z): break
    zs=np.array(zs)
    bad = np.where(~np.isfinite(zs) | (zs < -0.2))[0]
    first = bad[0] if len(bad) else -1
    print(f"  {obj:9} force={force:4.0f}N squeeze={squeeze_mm:3.1f}mm  peak={np.nanmax(zs)*1000:+7.1f} "
          f"final={zs[-1]*1000:+9.1f} mm  fails={first}/{len(path)}")
    if trace:
        print("    z trace:", " ".join(f"{zs[k]*1000:.0f}" for k in range(0,len(zs),12)))

for f, sq in ((10,3.0),(5,2.0),(3,1.5)):
    trial(f, sq, obj="test_tube", kp="body_grasp" if "body_grasp" in CATALOG["test_tube_16x100"].keypoints else "neck_grasp", width=0.016)
print()
for f, sq in ((10,3.0),(5,2.0)):
    trial(f, sq, obj="test_tube", kp="neck_grasp", width=0.016)
