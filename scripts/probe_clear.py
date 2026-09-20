"""Exactly how close is each part of the hand to the beaker, and where?

Reports, per hand body: its lowest point, that point's distance from the
beaker's axis, and the smallest clearance to the beaker's outer surface. A
negative clearance is real interpenetration; a small positive one is a part
hanging down BESIDE the glass, which looks like intersection from a three
quarter view and is not.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen"); sys.path.insert(0,"/mnt/c/Ujaan Docx/Research/labgen")
from arm_drive import configure_drives
from grasp import GraspFK, solve_grasp_ik, with_fingers, finger_q_for_gap, quat_to_matrix
from labgen.catalog import CATALOG
DOWN=np.array([0,0,-1.0])
R=CATALOG["beaker_250"].dims["outer_d"]/2; H=CATALOG["beaker_250"].dims["height"]
KPZ=CATALOG["beaker_250"].keypoints["rim_grasp"][2]
b=newton.ModelBuilder()
b.add_urdf("/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf",
  xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
  enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
  parse_visuals_as_colliders=False, mesh_maxhullvert=64)
lo=np.asarray(b.joint_limit_lower,float); hi=np.asarray(b.joint_limit_upper,float)
configure_drives(b,n_fingers=2,verbose=False)
m=b.finalize(); fk=GraspFK(m,slice(0,m.joint_coord_count))
lbl=[x.split('/')[-1] for x in b.body_label]
sb=m.shape_body.numpy(); sxf=m.shape_transform.numpy()
CX,CY=0.20,0.3464
q,_,_,_=solve_grasp_ik(fk,np.array([CX,CY,KPZ]),lo,hi,approach=DOWN)
full=np.zeros(8); full[:6]=q; full[6:]=finger_q_for_gap(0.073)
bodies=fk._eval(full)
print(f"beaker: axis at ({CX}, {CY}), r={R*1000:.0f} mm, top at z={H*1000:.0f} mm")
print(f"{'body':11} {'lowest z':>9} {'r there':>9} {'min clearance':>14}")
for i in range(m.shape_count):
    bi=int(sb[i])
    if bi < 5: continue
    g=m.shape_source[i]
    if g is None or not hasattr(g,'vertices'): continue
    v=np.asarray(g.vertices,float)[::3]
    xf=sxf[i]
    v=v@quat_to_matrix(xf[3:]).T+xf[:3]
    v=v@quat_to_matrix(bodies[bi,3:]).T+bodies[bi,:3]
    rad=np.linalg.norm(v[:,:2]-np.array([CX,CY]),axis=1)
    k=int(np.argmin(v[:,2]))
    band=(v[:,2]<H)&(v[:,2]>0.0)
    clr=(rad[band]-R).min()*1000 if band.any() else float('nan')
    print(f"{lbl[bi]:11} {v[k,2]*1000:8.1f} {rad[k]*1000:9.1f} {clr:13.2f} mm"
          f"{'   <-- INSIDE' if clr < 0 else ''}")
