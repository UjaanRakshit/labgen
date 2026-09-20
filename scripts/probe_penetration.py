"""How deep does the hand actually penetrate the beaker, in mm?

Treat the beaker as what it is -- a cylinder of radius 35 mm standing 95 mm
tall from its base -- and ask which gripper vertices are inside it, and by how
much. A number, so "still colliding" stops being a judgement call.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen"); sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, finger_q_for_gap, quat_to_matrix, N_ARM
from labgen.catalog import CATALOG

DOWN = np.array([0, 0, -1.0])
R_BEAKER = CATALOG["beaker_250"].dims["outer_d"] / 2
H_BEAKER = CATALOG["beaker_250"].dims["height"]

b = newton.ModelBuilder()
b.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
           xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
           parse_visuals_as_colliders=True, mesh_maxhullvert=64)
lo = np.asarray(b.joint_limit_lower, float); hi = np.asarray(b.joint_limit_upper, float)
configure_drives(b, n_fingers=2, verbose=False)
m = b.finalize(); fk = GraspFK(m, slice(0, m.joint_coord_count))
sb = m.shape_body.numpy(); sxf = m.shape_transform.numpy()

def hand_cloud(bodies, stride=3):
    out = []
    for i in range(m.shape_count):
        bi = int(sb[i])
        if bi < m.body_count - 3:     # gripper + two tips only
            continue
        g = m.shape_source[i]
        if g is None or not hasattr(g, "vertices"):
            continue
        v = np.asarray(g.vertices, float)[::stride]
        xf = sxf[i]
        v = v @ quat_to_matrix(xf[3:]).T + xf[:3]
        out.append(v @ quat_to_matrix(bodies[bi, 3:]).T + bodies[bi, :3])
    return np.vstack(out)

print(f"beaker: r={R_BEAKER*1000:.0f} mm, h={H_BEAKER*1000:.0f} mm")
print(f"{'keypoint':12} {'kp_z':>6} {'gap':>7} {'inside':>8} {'max_pen':>9}")
for kp in ("body_grasp", "rim_grasp"):
    kz = CATALOG["beaker_250"].keypoints[kp][2]
    target = np.array([0.20, 0.3464, kz])          # beaker base at z=0
    q, pos, ang, ok = solve_grasp_ik(fk, target, lo, hi, approach=DOWN)
    for gap in (0.070, 0.078, 0.086, 0.094):
        try:
            fq = finger_q_for_gap(gap)
        except Exception:
            continue
        full = np.zeros(8); full[:6] = q; full[6:] = fq
        bodies = fk._eval(full)
        P = hand_cloud(bodies)
        # cylinder axis is vertical through (0.20, 0.3464), base z=0
        radial = np.linalg.norm(P[:, :2] - np.array([0.20, 0.3464]), axis=1)
        inside = (radial < R_BEAKER) & (P[:, 2] > 0.0) & (P[:, 2] < H_BEAKER)
        pen = (R_BEAKER - radial[inside]).max() * 1000 if inside.any() else 0.0
        print(f"{kp:12} {kz*1000:6.1f} {gap*1000:7.1f} {int(inside.sum()):8d} "
              f"{pen:8.2f} mm")
