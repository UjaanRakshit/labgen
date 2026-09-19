"""Measure the real gripping pad out of the shipped tip STL.

The tip's convex hull spans +/-36 mm, so two of them at a 66 mm gap each
swallow a 70 mm beaker whole -- which is why the grasp launched it. The pad
that actually touches an object is a small flat face on the inner side. Measure
it from the mesh instead of inventing a box.
"""
import sys, numpy as np, trimesh
P="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/assets/tip_left.stl"
m = trimesh.load(P)
v = np.asarray(m.vertices)
print(f"tip_left.stl: {len(v)} verts  extent(mm)={np.round((v.max(0)-v.min(0))*1000,1)}")
print(f"  min={np.round(v.min(0)*1000,1)}  max={np.round(v.max(0)*1000,1)}")
print(f"  watertight={m.is_watertight}  volume={m.volume*1e6:.2f} cm^3")
# Which axis is the closing direction? In the URDF the fingers translate along
# the gripper's local X, which maps to world -Y at home.
for ax, name in enumerate("xyz"):
    hist, edges = np.histogram(v[:, ax], bins=12)
    print(f"  {name}: " + " ".join(f"{h:4d}" for h in hist))
    print(f"     edges(mm) " + " ".join(f"{e*1000:6.1f}" for e in edges[::3]))
# The pad: vertices in the last 20% of the closing axis, i.e. the inner face.
for ax in range(3):
    lo, hi = v[:,ax].min(), v[:,ax].max()
    for side, sel in (("max", v[v[:,ax] > hi - 0.1*(hi-lo)]), ("min", v[v[:,ax] < lo + 0.1*(hi-lo)])):
        if len(sel) < 20: continue
        ext = (sel.max(0)-sel.min(0))*1000
        print(f"  axis {'xyz'[ax]} {side}-face: {len(sel):5d} verts  extent {np.round(ext,1)} mm")
