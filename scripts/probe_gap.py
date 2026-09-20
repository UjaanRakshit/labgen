"""Measure the real pad-to-pad gap as a function of the finger joint value.

finger_q_for_gap was built from the separation of the tip BODY ORIGINS (89 mm
open, 5 mm closed). Those origins are joint locations and are 36 mm away from
the geometry -- the same error that put the carried object behind the hand. So
the gap model is derived from the wrong quantity too, and commanding "70 mm"
closes the pads somewhere inside the glass.

Measure it from the meshes: take the vertices of each tip that sit in the
region where an object is actually held, project them onto the closing axis,
and read off how far apart the innermost surfaces are.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, quat_to_matrix, N_ARM

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
b = newton.ModelBuilder()
b.add_urdf(URDF, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
           parse_visuals_as_colliders=True, mesh_maxhullvert=64)
configure_drives(b, n_fingers=2, verbose=False)
m = b.finalize()
fk = GraspFK(m, slice(0, m.joint_coord_count))
shape_body = m.shape_body.numpy(); shape_xf = m.shape_transform.numpy()
L, R = m.body_count - 2, m.body_count - 1

def tip_world(bodies, want):
    out = []
    for i in range(m.shape_count):
        if int(shape_body[i]) != want:
            continue
        g = m.shape_source[i]
        if g is None or not hasattr(g, "vertices"):
            continue
        v = np.asarray(g.vertices, float)
        xf = shape_xf[i]
        v = v @ quat_to_matrix(xf[3:]).T + xf[:3]
        v = v @ quat_to_matrix(bodies[want, 3:]).T + bodies[want, :3]
        out.append(v)
    return np.vstack(out)

print(f"{'q':>9} {'origins':>9} {'pads':>9}   (mm apart)")
rows = []
for q in np.linspace(0.0, -0.047, 12):
    jq = m.joint_q.numpy().copy(); jq[6:8] = q
    m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, fk.state)
    bodies = fk.state.body_q.numpy()
    gp = fk.grasp_point_from_state(bodies)
    axis = quat_to_matrix(bodies[-3, 3:]) @ np.array([0.0, 0.0, 1.0])   # approach
    close = bodies[L, :3] - bodies[R, :3]
    close = close / np.linalg.norm(close)                                # closing axis

    gaps = []
    for body, sign in ((L, +1.0), (R, -1.0)):
        v = tip_world(bodies, body) - gp
        along = v @ axis                       # along the approach direction
        lateral = v @ close                    # along the closing direction
        # only the material in the pad zone: within 30 mm of the grasp point
        # along the approach axis, and on its own side of the centreline.
        sel = (np.abs(along) < 0.030) & (sign * lateral > 0)
        if not sel.any():
            gaps.append(np.nan); continue
        gaps.append(sign * np.min(sign * lateral[sel]))
    pad_gap = gaps[0] - gaps[1]
    origin_gap = float(np.linalg.norm(bodies[L, :3] - bodies[R, :3]))
    rows.append((q, origin_gap, pad_gap))
    print(f"{q:9.4f} {origin_gap*1000:9.1f} {pad_gap*1000:9.1f}")

rows = np.array(rows)
ok = np.isfinite(rows[:, 2])
A = np.polyfit(rows[ok, 2], rows[ok, 0], 1)
print(f"\nlinear fit q(gap) = {A[0]:.6f} * gap + {A[1]:.6f}")
print(f"pad gap at q=0      : {rows[0,2]*1000:.1f} mm")
print(f"pad gap at q=-0.047 : {rows[-1,2]*1000:.1f} mm")
for want in (0.070, 0.016):
    q = np.polyval(A, want)
    print(f"  to hold {want*1000:.0f} mm -> q={q:.5f}"
          f"{'  (OUT OF RANGE)' if not (-0.047 <= q <= 0.0) else ''}")
