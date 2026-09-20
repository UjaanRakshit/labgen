"""Where do the finger pads actually sit in world space?

The carry used the midpoint of the two tip BODY origins as the grasp point.
Body origins are joint locations, not geometry: the tip STL spans z -219..-126
mm in its own file frame and the URDF <origin> shifts it somewhere else again.
So that midpoint can be nowhere near where the fingers close, which is what the
render shows -- the beaker hanging level with the wrist instead of between the
pads.

Measure it: transform each tip's collision mesh into world and look at where
the geometry really is.
"""
import sys, numpy as np, newton, warp as wp
sys.path.insert(0,"/home/ujaan/isaac/labgen")
from arm_drive import configure_drives
from grasp import quat_to_matrix, N_ARM
URDF="/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf"
b = newton.ModelBuilder()
b.add_urdf(URDF, xform=wp.transform(wp.vec3(0,0,0), wp.quat_identity()), floating=False,
           enable_self_collisions=False, collapse_fixed_joints=True, up_axis=newton.Axis.Z,
           parse_visuals_as_colliders=True, mesh_maxhullvert=64)
configure_drives(b, n_fingers=2, verbose=False)
m = b.finalize()
lbl = [x.split('/')[-1] for x in b.body_label]
sb = m.shape_body.numpy(); src = m.shape_source
xf = m.shape_transform.numpy()
s = m.state()
for fq in (0.0, -0.047):
    jq = m.joint_q.numpy().copy(); jq[6:8] = fq
    m.joint_q.assign(jq.astype(np.float32))
    newton.eval_fk(m, m.joint_q, m.joint_qd, s)
    bq = s.body_q.numpy()
    print(f"\n=== fingers q={fq} ===")
    mids = []
    for i in range(m.shape_count):
        bi = int(sb[i])
        if bi < 0 or lbl[bi] not in ("tip_left","tip_right","gripper"): continue
        g = src[i]
        v = np.asarray(g.vertices)
        # shape local transform, then body transform
        st = xf[i]
        Rs = quat_to_matrix(st[3:]); vs = v @ Rs.T + st[:3]
        Rb = quat_to_matrix(bq[bi,3:]); vw = vs @ Rb.T + bq[bi,:3]
        lo, hi = vw.min(0), vw.max(0)
        print(f"  {lbl[bi]:10} body_origin={np.round(bq[bi,:3],4)}")
        print(f"  {'':10} mesh world  min={np.round(lo,4)} max={np.round(hi,4)}")
        if lbl[bi] in ("tip_left","tip_right"):
            mids.append((lbl[bi], vw))
    if len(mids) == 2:
        (_, A), (_, B) = mids
        print(f"  midpoint of tip BODY ORIGINS : "
              f"{np.round(0.5*(bq[[i for i,l in enumerate(lbl) if l=='tip_left'][0],:3]+bq[[i for i,l in enumerate(lbl) if l=='tip_right'][0],:3]),4)}")
        print(f"  centroid of tip GEOMETRY     : {np.round(0.5*(A.mean(0)+B.mean(0)),4)}")
        # the pinch: lowest 15% in z of each pad, averaged
        def pad(v):
            z0 = v[:,2].min(); sel = v[v[:,2] < z0 + 0.15*(v[:,2].max()-z0)]
            return sel.mean(0)
        print(f"  pad-tip midpoint (lowest 15%): {np.round(0.5*(pad(A)+pad(B)),4)}")
