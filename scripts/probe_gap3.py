"""How far must the jaws open to CLEAR a cylinder of a given radius?

The previous calibration used the minimum distance between the two finger
meshes. That distance is measured between the pointed tips of the two wedges,
which are the closest features -- but the surface that meets a round object is
the flat inner face, set back from the tip. Setting "gap = 70 mm" by the tip
measure therefore still buries the faces in a 70 mm cylinder.

The question that actually matters is not finger-to-finger, it is
finger-to-object: for a cylinder standing on the grasp point, how close does
any jaw vertex come to its axis?
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

def cloud(bodies, want, stride=3):
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

# The arm's home pose has the approach along +X, so a "standing cylinder" here
# is along the world Y axis of the gripper frame. Use the gripper frame
# directly: the object axis is the gripper's local -Y (its "down").
print(f"{'q':>9} {'tip-gap':>9} {'clearance':>10}   (mm; clearance = min jaw distance from the object axis)")
rows = []
for q in np.linspace(0.0, -0.047, 20):
    jq = m.joint_q.numpy().copy(); jq[6:8] = q
    m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, fk.state)
    bodies = fk.state.body_q.numpy()
    gp = fk.grasp_point_from_state(bodies)
    Rg = quat_to_matrix(bodies[-3, 3:])
    approach = Rg @ np.array([0.0, 0.0, 1.0])
    obj_axis = Rg @ np.array([0.0, -1.0, 0.0])      # the object stands along this

    A, B = cloud(bodies, L), cloud(bodies, R)
    V = np.vstack([A, B]) - gp
    # jaw only, and within the object's height band
    V = V[(V @ approach > -0.005)]
    along = V @ obj_axis
    V = V[np.abs(along) < 0.050]
    radial = np.linalg.norm(V - np.outer(V @ obj_axis, obj_axis), axis=1)
    clearance = radial.min() if len(radial) else np.nan

    Ad = A[(A - gp) @ approach > -0.005]; Bd = B[(B - gp) @ approach > -0.005]
    tipgap = np.sqrt(((Ad[:, None, :] - Bd[None, :, :]) ** 2).sum(-1)).min()
    rows.append((q, tipgap, clearance))
    print(f"{q:9.4f} {tipgap*1000:9.2f} {clearance*1000:10.2f}")

rows = np.array(rows)
print("\nq needed to clear a given cylinder DIAMETER:")
for d in (0.016, 0.028, 0.070, 0.100):
    need = d / 2
    ok = rows[rows[:, 2] >= need]
    if len(ok) == 0:
        print(f"  {d*1000:5.0f} mm -> NOT POSSIBLE (max clearance "
              f"{rows[:,2].max()*2000:.0f} mm diameter)")
    else:
        q = ok[np.argmax(ok[:, 0])][0]          # least-open q that still clears
        print(f"  {d*1000:5.0f} mm -> q={q:+.5f}   (tip-gap there = "
              f"{np.interp(q, rows[::-1,0], rows[::-1,1])*1000:.1f} mm)")
