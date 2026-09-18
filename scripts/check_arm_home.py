"""Does the arm actually HOLD its documented home pose?

The YAM README states: "Home pose means all six arm joint coordinates are
zero", and at that pose the gripper mount frame sits at global
(0.110597512463, 0, 0.173501403591) metres in the base frame.

That is a number to measure against, not a vibe. T4's acceptance is the arm
standing at the robot base origin without intersecting the bench; if it sags
under gravity the render will still look like "an arm on a bench" while the
kinematics are wrong.
"""
import sys, math, numpy as np, newton, warp as wp

URDF = sys.argv[1]
DOC_HOME_GRIPPER = np.array([0.110597512463, 0.0, 0.173501403591])

b = newton.ModelBuilder()
b.default_joint_cfg = newton.ModelBuilder.JointDofConfig(
    target_ke=float(sys.argv[2]) if len(sys.argv) > 2 else 400.0,
    target_kd=float(sys.argv[3]) if len(sys.argv) > 3 else 40.0,
    effort_limit=float(sys.argv[4]) if len(sys.argv) > 4 else 10.0,
    armature=0.01)
xf = wp.transform(wp.vec3(0.0,0.0,0.0), wp.quat_identity())
if URDF.endswith(".xml"):
    b.add_mjcf(URDF, xform=xf, floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z)
else:
    b.add_urdf(URDF, xform=xf, floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z)
m = b.finalize()
s0, s1 = m.state(), m.state()
c = m.control()
# Populate body transforms from the joint configuration. Without this the
# bodies start at identity, the arm is drawn folded into the origin, and it
# then "falls" into its real pose on the first step -- which looks exactly like
# a collapsing arm and is nothing of the kind.
newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
newton.eval_fk(m, m.joint_q, m.joint_qd, s1)
ndof = m.joint_dof_count
tgt = c.joint_target_q.numpy().copy(); tgt[:] = 0.0; c.joint_target_q.assign(tgt)
solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)

q0 = s0.body_q.numpy().copy()
print(f"dof={ndof} bodies={m.body_count}")
el = m.joint_effort_limit.numpy() if hasattr(m, "joint_effort_limit") else None
print(f"effort limits in model  : {el[:8] if el is not None else 'n/a'}")
ke = m.joint_target_ke.numpy() if hasattr(m, "joint_target_ke") else None
print(f"target_ke in model      : {ke[:8] if ke is not None else 'n/a'}")
print(f"gripper body at t=0      : {q0[-1,:3]}")
print(f"documented home gripper  : {DOC_HOME_GRIPPER}")
print(f"initial error            : {np.linalg.norm(q0[-1,:3]-DOC_HOME_GRIPPER)*1000:.2f} mm")

dt = 1/240
for i in range(int(2.0*240)):
    contacts = m.collide(s0)
    solver.step(s0, s1, c, contacts, dt)
    s0, s1 = s1, s0
    if not np.isfinite(s0.body_q.numpy()).all():
        print(f"DIVERGED at {i}"); sys.exit(1)

q1 = s0.body_q.numpy()
jq = s0.joint_q.numpy() if hasattr(s0, "joint_q") else None
drift = np.linalg.norm(q1[:,:3]-q0[:,:3], axis=1)
print(f"\nafter 2 s holding target=0:")
print(f"gripper body              : {q1[-1,:3]}")
print(f"sag from t=0              : {np.linalg.norm(q1[-1,:3]-q0[-1,:3])*1000:.2f} mm")
print(f"worst body drift          : {drift.max()*1000:.2f} mm")
if jq is not None:
    print(f"joint coords (should be ~0): {np.round(jq[:6],4)}")
    print(f"max |joint| deviation     : {np.abs(jq[:6]).max():.4f} rad "
          f"({math.degrees(np.abs(jq[:6]).max()):.2f} deg)")
print("\nVERDICT:", "HOLDS home pose" if drift.max() < 0.002 else "SAGS -- gains insufficient")
