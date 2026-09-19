import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from arm_ik import ArmFK

b = newton.ModelBuilder(); b.add_usd("bench_5.usda")
sc, sd = b.joint_coord_count, b.joint_dof_count
a = newton.ModelBuilder()
a.add_mjcf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
lower = np.asarray(a.joint_limit_lower,float)[:6]; upper = np.asarray(a.joint_limit_upper,float)[:6]
b.add_builder(a)
coords = slice(sc, sc+a.joint_coord_count)
m = b.finalize()
fk = ArmFK(m, coords)

print("FK responds to each joint? (move joint j by +0.5 rad)")
h = np.zeros(6); p0 = fk.position(h)
print(f"  home -> {np.round(p0,4)}")
for j in range(6):
    q = h.copy(); q[j] = min(0.5, upper[j])
    p = fk.position(q)
    print(f"  j{j+1}={q[j]:+.2f} -> {np.round(p,4)}  moved {np.linalg.norm(p-p0)*1000:6.1f} mm")

print("\nreachable workspace (4000 random configs within limits):")
rng = np.random.default_rng(0)
pts = np.array([fk.position(rng.uniform(lower,upper)) for _ in range(4000)])
print(f"  x [{pts[:,0].min():+.3f}, {pts[:,0].max():+.3f}]")
print(f"  y [{pts[:,1].min():+.3f}, {pts[:,1].max():+.3f}]")
print(f"  z [{pts[:,2].min():+.3f}, {pts[:,2].max():+.3f}]")
r = np.linalg.norm(pts, axis=1)
print(f"  radius from base: [{r.min():.3f}, {r.max():.3f}] m")
tgt = np.array([-0.05, 0.28, 0.083])
d = np.linalg.norm(pts - tgt, axis=1)
print(f"\n  target {tgt} is {np.linalg.norm(tgt):.3f} m from base")
print(f"  closest sampled point to it: {d.min()*1000:.1f} mm at {np.round(pts[d.argmin()],4)}")
