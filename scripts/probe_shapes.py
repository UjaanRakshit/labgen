import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
b = newton.ModelBuilder()
b.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
print("bodies:", list(b.body_label))
print(f"shapes: {b.shape_count}")
m = b.finalize()
sb = m.shape_body.numpy()
lbl = list(b.body_label)
flags = m.shape_flags.numpy() if hasattr(m,'shape_flags') else None
src = m.shape_source
for i in range(m.shape_count):
    body = sb[i]
    name = lbl[body].split('/')[-1] if 0 <= body < len(lbl) else f"world({body})"
    g = src[i] if src is not None and i < len(src) else None
    nv = len(g.vertices) if g is not None and hasattr(g,'vertices') else '-'
    lo = m.shape_collision_aabb_lower.numpy()[i]; hi = m.shape_collision_aabb_upper.numpy()[i]
    print(f"  shape[{i}] body={name:12} verts={nv:>6}  flags={flags[i] if flags is not None else '?'}"
          f"  size={np.round(hi-lo,4)}")
missing = set(range(len(lbl))) - set(int(x) for x in sb)
print("\nbodies with NO collision shape:", [lbl[i].split('/')[-1] for i in sorted(missing)])
