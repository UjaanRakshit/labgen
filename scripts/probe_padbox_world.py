"""Find the finger pads in the WORLD frame, where "facing the other finger" means something.

scripts/probe_padbox.py found the pad by taking, in each tip body's OWN frame,
the extreme closest to y = 0 -- reasoning that the fingers meet there. That was
wrong: each tip has its own frame, and body-local y = 0 says nothing about
where the other finger is. The acceptance test showed the result -- the pad
boxes sat on the OUTER faces, 18 mm above the beaker rim, and moved APART as the
jaws closed (42.7 mm apart open, 63.7 mm closed). The grasp could never work.

This does it properly:
  1. put the arm in a real grasp configuration, fingers at a known opening
  2. transform both tip meshes into the world frame
  3. the closing axis is the finger prismatic joint's world axis
  4. the jaw face of each tip is its vertices nearest the OTHER tip along that
     axis; the contact pad is the DISTAL part of that face, i.e. furthest along
     the approach axis (lowest, with the gripper pointing down)
  5. report the pad box back in each tip's own body frame, so it can be
     authored as a collider, and check it behaves: pads must get CLOSER as the
     jaws close, and sit at the fingertips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasp import quat_to_matrix, set_state                               # noqa: E402
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.control import YAM_JAWS                                       # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
# The rig's recorded reset pose: gripper points down over the bench.
READY = np.array([0.0, 1.5886, 0.9016, 1.0007, 0.6594, 0.0])
FACE_BAND_M = 0.004      # vertices within 4 mm of the innermost plane
DISTAL_BAND_M = 0.020    # and within 20 mm of the fingertip along the approach axis


def main() -> int:
    b = newton.ModelBuilder()
    b.add_urdf(URDF, xform=wp.transform(wp.vec3(0, 0, 0), wp.quat_identity()),
               floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=True, mesh_maxhullvert=64)
    model = b.finalize()
    tips = (model.body_count - 2, model.body_count - 1)       # tip_left, tip_right
    mount = model.body_count - 3
    shape_body = model.shape_body.numpy()
    shape_xf = model.shape_transform.numpy()

    def tip_vertices_local(bi):
        out = []
        for i in range(model.shape_count):
            if int(shape_body[i]) != bi:
                continue
            geo = model.shape_source[i]
            if geo is None or not hasattr(geo, "vertices"):
                continue
            v = np.asarray(geo.vertices, float)
            xf = shape_xf[i]
            out.append(v @ quat_to_matrix(xf[3:]).T + xf[:3])
        return np.vstack(out)

    local = {bi: tip_vertices_local(bi) for bi in tips}

    def at_gap(gap):
        s = model.state()
        q = np.concatenate([READY, [YAM_JAWS.q_for_gap(gap)] * 2]).astype(np.float32)
        set_state(model, s, q)
        bq = s.body_q.numpy()
        world = {bi: local[bi] @ quat_to_matrix(bq[bi, 3:]).T + bq[bi, :3] for bi in tips}
        return bq, world

    bq, world = at_gap(0.060)
    approach = quat_to_matrix(bq[mount, 3:]) @ np.array([0.0, 0.0, 1.0])
    print(f"approach axis (gripper +Z) in world: {np.round(approach, 3)}  "
          f"({'pointing down' if approach[2] < -0.9 else 'NOT down -- check the pose'})")

    c = {bi: world[bi].mean(axis=0) for bi in tips}
    close_axis = c[tips[1]] - c[tips[0]]
    close_axis -= approach * (close_axis @ approach)     # keep it perpendicular
    close_axis /= np.linalg.norm(close_axis)

    pads_local = {}
    for bi, other, sign in ((tips[0], tips[1], 1.0), (tips[1], tips[0], -1.0)):
        w = world[bi]
        # FINGERTIP FIRST. On an L-shaped finger the surface nearest the other
        # finger is the bracket by the mount, which moves OPPOSITE to the jaw --
        # picking the face first and the distal end second finds that bracket
        # and gives pads that separate as the jaws close. i2rt's own note: only
        # the distal half of the finger is the jaw.
        depth = w @ approach                          # larger = further toward the fingertip
        tipzone = w[depth >= depth.max() - DISTAL_BAND_M]
        s_along = tipzone @ (sign * close_axis)       # larger = toward the other tip
        pad = tipzone[s_along >= s_along.max() - FACE_BAND_M]
        face = pad
        # back into this tip's body frame
        R = quat_to_matrix(bq[bi, 3:])
        pl = (pad - bq[bi, :3]) @ R
        pads_local[bi] = pl
        lo, hi = pl.min(axis=0), pl.max(axis=0)
        print(f"tip body {bi}: jaw face {len(face)} verts, distal pad {len(pad)} verts  "
              f"box {np.round((hi-lo)*1000, 1)} mm  centre {np.round((hi+lo)/2*1000, 2)} mm (body frame)")

    # Verify behaviour across the stroke using the pads found above.
    print()
    print(f"{'jaw gap':>8} {'pad-pad dist':>13} {'pad z (world)':>15}   must SHRINK as the gap shrinks")
    for gap in (0.090, 0.070, 0.050, 0.030):
        bq2, _ = at_gap(gap)
        centres = []
        for bi in tips:
            pl = pads_local[bi]
            ctr = (pl.min(axis=0) + pl.max(axis=0)) / 2
            centres.append(bq2[bi, :3] + quat_to_matrix(bq2[bi, 3:]) @ ctr)
        d = np.linalg.norm(centres[0] - centres[1])
        print(f"{gap*1000:6.0f}mm {d*1000:11.1f}mm {centres[0][2]*1000:10.1f} mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
