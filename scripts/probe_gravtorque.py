"""Is the 20 N.m shoulder figure a fact about the arm, or about my controller?

Claimed earlier: holding a 0.40 m top-down pose needs ~20 N.m against the YAM's
rated 10, and read as a possible hardware limit. But that was measured as "what
a pure position-PD needs to hold station", and a PD with no gravity feedforward
must supply the ENTIRE gravity load through position error. A real arm
controller compensates gravity, so the two numbers are not the same quantity.

Computes the actual static gravity torque about each joint axis, which is what
the hardware has to produce.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, quat_to_matrix, N_ARM
DOWN = np.array([0, 0, -1.0])
URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"

b = newton.ModelBuilder()
b.add_urdf(URDF, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
lo = np.asarray(b.joint_limit_lower, float); hi = np.asarray(b.joint_limit_upper, float)
masses = np.asarray(b.body_mass, float)
labels = [x.split("/")[-1] for x in b.body_label]
print("link masses from the URDF:")
for n, mm in zip(labels, masses):
    print(f"   {n:12} {mm*1000:7.1f} g")
print(f"   {'TOTAL':12} {masses.sum()*1000:7.1f} g  ({masses.sum():.3f} kg)")

configure_drives(b, n_fingers=2, verbose=False)
m = b.finalize()
fk = GraspFK(m, slice(0, m.joint_coord_count))

# Joint axes and anchors, in world, at a given configuration.
ja = np.asarray(b.joint_axis, float) if hasattr(b, "joint_axis") else None

def gravity_torques(q):
    """Static torque each revolute joint must hold, from first principles.

    tau_i = sum over links OUTBOARD of joint i of  (r_link - r_joint) x (m g)
    projected on the joint axis. No solver involved.
    """
    bodies = fk._eval(np.concatenate([q, [0.0, 0.0]]))
    g = np.array([0.0, 0.0, -9.81])
    taus = []
    for i in range(N_ARM):
        # joint i sits at body i's origin; everything from i outward loads it
        anchor = bodies[i, :3]
        R = quat_to_matrix(bodies[i, 3:])
        # revolute axis in world: take it from the child frame's local axis
        axis = R @ (np.asarray(ja[i][:3], float) if ja is not None else np.array([0, 1, 0.]))
        axis = axis / (np.linalg.norm(axis) or 1.0)
        t = np.zeros(3)
        for j in range(i, len(masses)):
            t += np.cross(bodies[j, :3] - anchor, masses[j] * g)
        taus.append(abs(float(t @ axis)))
    return np.array(taus)

for label, target in (("beaker grasp, 0.40 m reach", np.array([0.20, 0.3464, 0.083])),
                      ("full extension, 0.60 m",     np.array([0.00, 0.6000, 0.150])),
                      ("compact, 0.25 m",            np.array([0.15, 0.2000, 0.200]))):
    q, pos, ang, ok = solve_grasp_ik(fk, target, lo, hi, approach=DOWN)
    if not ok:
        print(f"\n{label}: IK did not converge, skipping")
        continue
    taus = gravity_torques(q[:N_ARM])
    print(f"\n{label}  (reach {np.linalg.norm(target):.3f} m)")
    print("   static gravity torque per joint (N.m): " +
          " ".join(f"j{i+1}={t:5.2f}" for i, t in enumerate(taus)))
    print(f"   worst = {taus.max():.2f} N.m at j{int(taus.argmax())+1}"
          f"   vs the YAM's rated 10 N.m  ->  "
          f"{'WITHIN spec' if taus.max() <= 10 else 'EXCEEDS spec'}")
