"""Can the fingers actually pick the beaker up? Three collider strategies.

The tip STL is an L-shaped part. Its convex HULL fills the space between the
fingers, so each finger swallows a 70 mm beaker and the solver ejects it.
Decomposition and SDF both preserve concavity; this measures which one lets a
grasp work, by the only metric that matters -- does the beaker come off the
bench and stay in the hand.
"""
import sys, inspect, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from arm_ik import lerp_path
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
print("approximate_meshes:", inspect.signature(newton.ModelBuilder.approximate_meshes))
DOWN=np.array([0,0,-1.0]); FPS=60; DT=1/240; SUB=4
scene = SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
URDF="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"

def build(mode):
    b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
    sc, sd, n0 = b.joint_coord_count, b.joint_dof_count, b.shape_count
    a = newton.ModelBuilder()
    if mode == "sdf":
        a.default_shape_cfg.force_sdf = True
        a.default_shape_cfg.sdf_target_voxel_size = 0.002
    a.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
               enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=True, mesh_maxhullvert=64)
    if mode == "coacd":
        try:
            a.approximate_meshes(method="coacd")
        except TypeError:
            a.approximate_meshes("coacd")
    lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
    b.add_builder(a)
    coords=slice(sc,sc+a.joint_coord_count); dofs=slice(sd,sd+a.joint_dof_count)
    configure_drives(b, dof_slice=dofs, n_fingers=2, verbose=False)
    return b.finalize(), lo, hi, coords, r

def trial(mode):
    try:
        m, lo, hi, coords, r = build(mode)
    except Exception as e:
        print(f"  {mode:8} BUILD FAILED: {type(e).__name__}: {str(e)[:70]}"); return
    paths={v:k for k,v in r["path_body_map"].items()}
    bk=[i for i in range(4) if paths[i].endswith("beaker")][0]
    fk = GraspFK(m, coords)
    g = np.asarray(scene.by_id("beaker").position)+np.asarray(CATALOG["beaker_250"].keypoints["body_grasp"])
    qa,_,_,_ = solve_grasp_ik(fk, g+[0,0,0.14], lo, hi, approach=DOWN)
    qg,_,_,_ = solve_grasp_ik(fk, g, lo, hi, approach=DOWN, seed=qa)
    gq = finger_q_for_gap(0.066)
    path = ([with_fingers(qa,0.0)]*20
            + lerp_path([with_fingers(qa,0.0), with_fingers(qg,0.0)],[1.0],FPS)
            + lerp_path([with_fingers(qg,0.0), with_fingers(qg,gq)],[0.8],FPS)
            + [with_fingers(qg,gq)]*20
            + lerp_path([with_fingers(qg,gq), with_fingers(qa,gq)],[1.2],FPS)
            + [with_fingers(qa,gq)]*40)
    s0,s1=m.state(),m.state(); c=m.control()
    q=m.joint_q.numpy().copy(); q[coords]=with_fingers(qa,0.0); m.joint_q.assign(q.astype(np.float32))
    newton.eval_fk(m,m.joint_q,m.joint_qd,s0); newton.eval_fk(m,m.joint_q,m.joint_qd,s1)
    solver=newton.solvers.SolverMuJoCo(m,iterations=120,ls_iterations=60)
    tg=c.joint_target_q.numpy().copy()
    z0 = s0.body_q.numpy()[bk,2]; zmax=z0
    for cfg in path:
        tg[coords]=cfg; c.joint_target_q.assign(tg)
        for _ in range(SUB):
            ct=m.collide(s0); solver.step(s0,s1,c,ct,DT); s0,s1=s1,s0
        bq=s0.body_q.numpy()
        if not np.isfinite(bq).all(): print(f"  {mode:8} DIVERGED"); return
        zmax=max(zmax, bq[bk,2])
    zf = s0.body_q.numpy()[bk,2]
    verdict = "LIFTED and held" if zf>0.05 else ("lifted then dropped" if zmax>0.05 else "never left the bench")
    print(f"  {mode:8} z0={z0*1000:+7.1f} peak={zmax*1000:+7.1f} final={zf*1000:+8.1f} mm -> {verdict}")

for mode in ("hull", "coacd", "sdf"):
    trial(mode)
