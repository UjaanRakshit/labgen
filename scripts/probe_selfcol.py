import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
DOWN=np.array([0,0,-1.0])
scene = SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")

def run(self_col, label):
    b = newton.ModelBuilder(); b.add_usd("bench_arm.usda")
    sc, sd = b.joint_coord_count, b.joint_dof_count
    a = newton.ModelBuilder()
    a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
        xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
        enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
        parse_visuals_as_colliders=self_col, mesh_maxhullvert=64)
    lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
    b.add_builder(a)
    coords=slice(sc,sc+a.joint_coord_count); dofs=slice(sd,sd+a.joint_dof_count)
    configure_drives(b, dof_slice=dofs, verbose=False)
    m = b.finalize()
    lbls = list(m.shape_label)
    arm_shapes = {i for i,l in enumerate(lbls) if not l.startswith('/World')}
    fk = GraspFK(m, coords)
    grasp = np.asarray(scene.by_id("beaker").position)+np.asarray(CATALOG["beaker_250"].keypoints["body_grasp"])
    q,_,_,_ = solve_grasp_ik(fk, grasp, lo, hi, approach=DOWN)
    s0,s1=m.state(),m.state(); c=m.control()
    qa=m.joint_q.numpy().copy(); qa[coords]=with_fingers(q, finger_q_for_gap(0.060))
    m.joint_q.assign(qa.astype(np.float32))
    newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
    solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
    tg=c.joint_target_q.numpy().copy(); tg[coords]=with_fingers(q, finger_q_for_gap(0.060)); c.joint_target_q.assign(tg)
    for _ in range(20):
        ct=m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1=s1,s0
    n=int(ct.rigid_contact_count.numpy()[0])
    a0=ct.rigid_contact_shape0.numpy(); a1=ct.rigid_contact_shape1.numpy()
    arm_ct = sum(1 for k in range(min(n,len(a0)))
                 if int(a0[k]) in arm_shapes or int(a1[k]) in arm_shapes)
    fl = m.shape_flags.numpy()
    armfl = sorted({int(fl[i]) for i in arm_shapes})
    print(f"  {label:38} total={n:4d}  arm contacts={arm_ct}  arm flags={armfl}")

run(False, "parse_visuals_as_colliders=False")
run(True,  "parse_visuals_as_colliders=True")
