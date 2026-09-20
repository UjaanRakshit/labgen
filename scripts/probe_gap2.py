"""Gap between the finger meshes vs joint value -- measured, not assumed.

The body origins move APART as q goes 0 -> -0.047 (89 mm -> 5 mm lateral), but
the wedge-shaped tips extend inward from those origins, so the geometry does the
opposite: at q=0 the fingers touch, at q=-0.047 they are wide open. The old
model read the origins and got the whole thing backwards, which is why "close
to 70 mm" drove the pads almost shut and through the glass.

Minimum distance between the two tip point clouds, subsampled.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, quat_to_matrix

b = newton.ModelBuilder()
b.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
           xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
           parse_visuals_as_colliders=True, mesh_maxhullvert=64)
configure_drives(b, n_fingers=2, verbose=False)
m = b.finalize()
fk = GraspFK(m, slice(0, m.joint_coord_count))
sb = m.shape_body.numpy(); sxf = m.shape_transform.numpy()
L, R = m.body_count - 2, m.body_count - 1

def cloud(bodies, want, stride=7):
    out = []
    for i in range(m.shape_count):
        if int(sb[i]) != want:
            continue
        g = m.shape_source[i]
        if g is None or not hasattr(g, "vertices"):
            continue
        v = np.asarray(g.vertices, float)[::stride]
        xf = sxf[i]
        v = v @ quat_to_matrix(xf[3:]).T + xf[:3]
        out.append(v @ quat_to_matrix(bodies[want, 3:]).T + bodies[want, :3])
    return np.vstack(out)

print(f"{'q':>9} {'gap(mm)':>9}")
rows = []
for q in np.linspace(0.0, -0.047, 16):
    jq = m.joint_q.numpy().copy(); jq[6:8] = q
    m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, fk.state)
    bodies = fk.state.body_q.numpy()
    gp = fk.grasp_point_from_state(bodies)
    axis = quat_to_matrix(bodies[-3, 3:]) @ np.array([0.0, 0.0, 1.0])
    A, B = cloud(bodies, L), cloud(bodies, R)
    # Only the JAW: material in the distal half of the finger, i.e. at or
    # beyond the grasp point along the approach axis. The global minimum over
    # the whole mesh is dominated by the brackets near the mount, which sit a
    # fixed ~19 mm apart no matter what the jaws are doing.
    A = A[(A - gp) @ axis > -0.005]
    B = B[(B - gp) @ axis > -0.005]
    if len(A) == 0 or len(B) == 0:
        rows.append((q, np.nan)); print(f"{q:9.4f}      n/a"); continue
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
    rows.append((q, d))
    print(f"{q:9.4f} {d*1000:9.2f}")

rows = np.array(rows)
print(f"\nq=0      -> {rows[0,1]*1000:.1f} mm   q=-0.047 -> {rows[-1,1]*1000:.1f} mm")
# monotone? fit q(gap) over the usable span
order = np.argsort(rows[:, 1])
print("monotonic in q:", bool(np.all(np.diff(rows[:, 1]) >= -1e-6)))
for want in (0.070, 0.016, 0.100):
    q = float(np.interp(want, rows[order, 1], rows[order, 0]))
    inrange = rows[:, 1].min() <= want <= rows[:, 1].max()
    print(f"  gap {want*1000:5.0f} mm -> q={q:+.5f}{'' if inrange else '   (BEYOND THE FINGERS RANGE)'}")
