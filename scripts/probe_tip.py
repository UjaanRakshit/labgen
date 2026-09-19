import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
ARM="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml"

def run(z_off, label):
    b = newton.ModelBuilder(); r = b.add_usd("bench_arm.usda")
    sc, sd = b.joint_coord_count, b.joint_dof_count
    a = newton.ModelBuilder()
    a.add_mjcf(ARM, xform=wp.transform(wp.vec3(0,0,z_off), wp.quat_identity()),
        floating=False, enable_self_collisions=False,
        collapse_fixed_joints=True, up_axis=newton.Axis.Z)
    b.add_builder(a)
    configure_drives(b, dof_slice=slice(sd, sd+a.joint_dof_count), verbose=False)
    m = b.finalize()
    paths = {v:k for k,v in r["path_body_map"].items()}
    s0,s1 = m.state(), m.state(); c = m.control()
    newton.eval_fk(m, m.joint_q, m.joint_qd, s0); newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
    t = c.joint_target_q.numpy().copy(); t[sc:] = 0.0; c.joint_target_q.assign(t)
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    q0 = s0.body_q.numpy().copy()
    for _ in range(int(1.5*240)):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1 = s1,s0
        if not np.isfinite(s0.body_q.numpy()).all(): print(f"{label}: DIVERGED"); return
    q1 = s0.body_q.numpy()
    out=[]
    for i in range(4):
        n = paths.get(i,'?').replace('/World/','')
        dpos = np.linalg.norm(q1[i,:3]-q0[i,:3])*1000
        dquat = np.linalg.norm(q1[i,3:]-q0[i,3:])
        out.append(f"{n}={dpos:.1f}mm/rot{dquat:.3f}")
    print(f"{label}: " + "  ".join(out))

run(0.000, "arm at z=0.000")
run(0.002, "arm at z=0.002")
run(0.010, "arm at z=0.010")
