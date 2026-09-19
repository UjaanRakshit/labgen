"""Watch one grasp closely: commanded vs actual fingers, and the beaker's z."""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from arm_ik import lerp_path
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap, N_ARM
from labgen.catalog import CATALOG
from labgen.types import SceneSpec

DOWN = np.array([0,0,-1.0]); FPS=60; DT=1/240; SUB=4
scene = SceneSpec.read("/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json")
b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
sc, sd = b.joint_coord_count, b.joint_dof_count
a = newton.ModelBuilder()
a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
    parse_visuals_as_colliders=True, mesh_maxhullvert=64)
lo = np.asarray(a.joint_limit_lower,float); hi = np.asarray(a.joint_limit_upper,float)
b.add_builder(a)
coords = slice(sc, sc+a.joint_coord_count); dofs = slice(sd, sd+a.joint_dof_count)
configure_drives(b, dof_slice=dofs, verbose=False)
m = b.finalize()
paths = {v:k for k,v in r["path_body_map"].items()}
bk = [i for i in range(4) if paths[i].endswith("beaker")][0]
fk = GraspFK(m, coords)

obj = scene.by_id("beaker")
grasp = np.asarray(obj.position) + np.asarray(CATALOG["beaker_250"].keypoints["body_grasp"])
print("beaker body_grasp:", np.round(grasp,4), " outer_d=0.070")
q_app,_,_,_ = solve_grasp_ik(fk, grasp+[0,0,0.14], lo, hi, approach=DOWN)
q_at ,_,_,_ = solve_grasp_ik(fk, grasp, lo, hi, approach=DOWN, seed=q_app)
print("fingertip at grasp cfg:", np.round(fk.grasp_point(q_at),4))
gq = finger_q_for_gap(0.066)
print(f"finger target for 66mm gap: {gq:.5f}")

path = ([with_fingers(q_app,0.0)]*30 + lerp_path([with_fingers(q_app,0.0), with_fingers(q_at,0.0)],[1.0],FPS)
      + lerp_path([with_fingers(q_at,0.0), with_fingers(q_at,gq)],[0.8],FPS) + [with_fingers(q_at,gq)]*30
      + lerp_path([with_fingers(q_at,gq), with_fingers(q_app,gq)],[1.2],FPS) + [with_fingers(q_app,gq)]*60)

s0,s1 = m.state(), m.state(); c = m.control()
qa = m.joint_q.numpy().copy(); qa[coords] = with_fingers(q_app,0.0)
m.joint_q.assign(qa.astype(np.float32))
newton.eval_fk(m, m.joint_q, m.joint_qd, s0); newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
solver = newton.solvers.SolverMuJoCo(m, iterations=120, ls_iterations=60)
tg = c.joint_target_q.numpy().copy()
print("\n frame  cmd_fing  act_fing   beaker_z   tip_gap   contacts")
for i,cfg in enumerate(path):
    tg[coords] = cfg; c.joint_target_q.assign(tg)
    for _ in range(SUB):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,DT); s0,s1=s1,s0
    if i % 25 == 0 or i == len(path)-1:
        jq = s0.joint_q.numpy()
        bq = s0.body_q.numpy()
        gap = np.linalg.norm(bq[-1,:3]-bq[-2,:3])
        n_ct = int(ct.rigid_contact_count.numpy()[0]) if hasattr(ct,'rigid_contact_count') else -1
        print(f" {i:5d}  {cfg[6]:+.4f}  {jq[coords][6]:+.4f}   {bq[bk,2]*1000:+7.2f}mm  {gap*1000:6.1f}mm  {n_ct}")
