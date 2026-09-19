"""Where can the fingertips actually go with the wrist pointing DOWN?

Planning into poses that do not exist is what made the tasks fail: the IK hit
the position and came back 90 degrees off on the approach axis, the arm went
somewhere else, and the carried object was released into mid-air. Map the set
first, then only plan inside it.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik
DOWN=np.array([0,0,-1.0])
b = newton.ModelBuilder(); b.add_usd("bench_arm.usda")
sc, sd = b.joint_coord_count, b.joint_dof_count
a = newton.ModelBuilder()
a.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
    xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
    enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z)
lo=np.asarray(a.joint_limit_lower,float); hi=np.asarray(a.joint_limit_upper,float)
b.add_builder(a)
coords=slice(sc,sc+a.joint_coord_count)
configure_drives(b, dof_slice=slice(sd,sd+a.joint_dof_count), n_fingers=2, verbose=False)
m=b.finalize(); fk=GraspFK(m,coords)
print(f"{'z (mm)':>7} " + " ".join(f"{r*100:5.0f}" for r in (0.30,0.34,0.38,0.42,0.46,0.50)))
print("        " + " ".join("  cm " for _ in range(6)))
for z in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35):
    row=[]
    for rad in (0.30,0.34,0.38,0.42,0.46,0.50):
        okn=0
        for deg in (-50,-25,0,25,50):
            t=np.array([rad*np.sin(np.radians(deg)), rad*np.cos(np.radians(deg)), z])
            _,pos,ang,ok = solve_grasp_ik(fk,t,lo,hi,approach=DOWN,restarts=6)
            okn += int(ok)
        row.append(f"{okn}/5")
    print(f"{z*1000:7.0f} " + " ".join(f"{c:>5}" for c in row))
