"""Measure the finger PAD from the tip mesh, so a box collider can replace the hull.

The convex hull of a whole fingertip spans +/-36 mm because the part is an
L-shaped wedge: the bracket near the mount is far from the jaw, and a hull
bridges the two. Two hulls therefore engulf a 70 mm beaker and the solver
ejects it at speed -- measured at 9.7 m of displacement.

The jaw itself is only the distal slab. This finds it: take the tip mesh in
body-local coordinates, keep the vertices in the distal fraction along the
finger's long axis, and report that sub-volume's oriented extent. That is the
box worth authoring.

These numbers come from the URDF's own visual mesh. They are NOT caliper
measurements of the real pads, and everything downstream must say so.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

import newton

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grasp import quat_to_matrix                                          # noqa: E402

URDF = "/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"


def main() -> int:
    b = newton.ModelBuilder()
    b.add_urdf(URDF, xform=wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()),
               floating=False, enable_self_collisions=False,
               collapse_fixed_joints=True, up_axis=newton.Axis.Z,
               parse_visuals_as_colliders=True, mesh_maxhullvert=64)
    model = b.finalize()
    shape_body = model.shape_body.numpy()
    shape_xf = model.shape_transform.numpy()
    tips = (model.body_count - 1, model.body_count - 2)

    for bi in sorted(tips):
        pts = []
        for i in range(model.shape_count):
            if int(shape_body[i]) != bi:
                continue
            geo = model.shape_source[i]
            if geo is None or not hasattr(geo, "vertices"):
                continue
            v = np.asarray(geo.vertices, float)
            xf = shape_xf[i]
            pts.append(v @ quat_to_matrix(xf[3:]).T + xf[:3])
        if not pts:
            print(f"body {bi}: no mesh")
            continue
        v = np.vstack(pts)
        lo, hi = v.min(axis=0), v.max(axis=0)
        print(f"\nbody {bi}: {len(v)} verts")
        print(f"  full extent (mm): "
              f"x {lo[0]*1000:7.1f}..{hi[0]*1000:7.1f}  "
              f"y {lo[1]*1000:7.1f}..{hi[1]*1000:7.1f}  "
              f"z {lo[2]*1000:7.1f}..{hi[2]*1000:7.1f}")
        # The jaw is the slab FACING THE OTHER FINGER, not the distal end of
        # the longest axis -- the mesh is the whole L-shaped bracket and its
        # long axis runs down the finger, so slicing that way keeps the whole
        # 68 x 72 mm cross-section and isolates nothing.
        #
        # The two fingers separate along y (body 6 spans y<=3.3, body 7 y>=-3.3)
        # and at the closed configuration the jaw faces touch. So the pad is the
        # thin slab nearest y=0 on each finger.
        # The INNER face is whichever extreme lies closest to y = 0, because the
        # two fingers meet there when closed. Picking the larger extreme
        # instead measures the back of the bracket, which is a real surface
        # and completely the wrong one.
        face = hi[1] if abs(hi[1]) < abs(lo[1]) else lo[1]
        for slab_mm in (4.0, 8.0, 12.0):
            slab = slab_mm / 1000.0
            sel = v[np.abs(v[:, 1] - face) <= slab]
            if len(sel) < 20:
                continue
            slo, shi = sel.min(axis=0), sel.max(axis=0)
            ext = (shi - slo) * 1000
            ctr = (shi + slo) / 2.0
            print(f"  jaw slab within {slab_mm:4.1f} mm of the face: "
                  f"{len(sel):5d} verts  pad {ext[0]:6.1f} (w) x {ext[2]:6.1f} (l) "
                  f"x {ext[1]:5.1f} (t) mm  centre ({ctr[0]*1000:6.1f},"
                  f"{ctr[1]*1000:6.1f},{ctr[2]*1000:6.1f}) mm")
        # And the contact-relevant part: the distal half of that slab.
        sel = v[np.abs(v[:, 1] - face) <= 0.008]
        zc = sel[:, 2].max() - 0.5 * (sel[:, 2].max() - sel[:, 2].min())
        pad = sel[sel[:, 2] >= zc]
        if len(pad) > 20:
            plo, phi = pad.min(axis=0), pad.max(axis=0)
            ext = (phi - plo) * 1000
            ctr = (phi + plo) / 2.0
            print(f"  -> DISTAL HALF of the 8 mm slab: pad "
                  f"{ext[0]:6.1f} (w) x {ext[2]:6.1f} (l) x {ext[1]:5.1f} (t) mm"
                  f"  centre ({ctr[0]*1000:6.1f},{ctr[1]*1000:6.1f},"
                  f"{ctr[2]*1000:6.1f}) mm")
    return 0


if __name__ == "__main__":
    sys.exit(main())
