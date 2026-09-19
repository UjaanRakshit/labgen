import sys, math, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives, report

SRC = sys.argv[1]; KE = float(sys.argv[2]); KD = float(sys.argv[3]); EFF = float(sys.argv[4])
b = newton.ModelBuilder()
xf = wp.transform(wp.vec3(0,0,0), wp.quat_identity())
kw = dict(xform=xf, floating=False, enable_self_collisions=False,
          collapse_fixed_joints=True, up_axis=newton.Axis.Z)
(b.add_mjcf if SRC.endswith(".xml") else b.add_urdf)(SRC, **kw)
configure_drives(b, ke=KE, kd=KD, effort=EFF, verbose=False)
m = b.finalize()
report(m, "")
s0, s1 = m.state(), m.state(); c = m.control()
newton.eval_fk(m, m.joint_q, m.joint_qd, s0); newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
t = c.joint_target_q.numpy().copy(); t[:] = 0.0; c.joint_target_q.assign(t)
solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
q0 = s0.body_q.numpy().copy()
for i in range(480):
    ct = m.collide(s0); solver.step(s0, s1, c, ct, 1/240); s0, s1 = s1, s0
    if not np.isfinite(s0.body_q.numpy()).all(): print("DIVERGED"); sys.exit(1)
jq = s0.joint_q.numpy()
drift = np.linalg.norm(s0.body_q.numpy()[:,:3]-q0[:,:3], axis=1)
dev = float(np.abs(jq[:6]).max())
print(f"   max joint deviation: {dev:.4f} rad ({math.degrees(dev):.2f} deg)   "
      f"worst body drift: {drift.max()*1000:.2f} mm")
print("   VERDICT:", "HOLDS" if math.degrees(dev) < 2.0 else "sags")
