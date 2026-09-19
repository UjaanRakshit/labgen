import sys, numpy as np, newton, warp as wp
sys.path.insert(0, "/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, quat_to_matrix, N_ARM
U="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
b = newton.ModelBuilder()
b.add_urdf(U, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
configure_drives(b, verbose=False)
m = b.finalize()
fk = GraspFK(m, slice(0, m.joint_coord_count))
h = np.zeros(N_ARM)
pt, ax = fk.pose(h)
print("README says: at home the gripper +Z (approach) points along world +X")
print(f"  grasp point   : {np.round(pt,4)}")
print(f"  approach axis : {np.round(ax,4)}   <- expect ~[1, 0, 0]")
bodies = fk._eval(h)
print(f"  gripper quat  : {np.round(bodies[-3,3:],4)}")
R = quat_to_matrix(bodies[-3,3:])
for k,v in (("local X", [1,0,0]), ("local Y", [0,1,0]), ("local Z", [0,0,1])):
    print(f"  {k} in world : {np.round(R @ np.array(v),4)}")
print("\nCan the arm point DOWN at all? sampling 6000 configs:")
lo = np.asarray(b.joint_limit_lower,float)[:6]; hi = np.asarray(b.joint_limit_upper,float)[:6]
rng = np.random.default_rng(0)
best = (None, 999)
zs = []
for _ in range(6000):
    q = rng.uniform(lo,hi)
    a = fk.approach_axis(q)
    zs.append(a[2])
    ang = np.degrees(np.arccos(np.clip(a @ np.array([0,0,-1.0]),-1,1)))
    if ang < best[1]: best = (q, ang)
zs = np.array(zs)
print(f"  approach-axis z component: min={zs.min():+.3f} max={zs.max():+.3f}")
print(f"  best alignment with straight-down: {best[1]:.1f} deg")
