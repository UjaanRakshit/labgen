import sys, math, numpy as np, newton, warp as wp
from newton.viewer import ViewerGL
from PIL import Image
b = newton.ModelBuilder(); b.add_usd("bench_5.usda"); m = b.finalize()
s0 = m.state(); newton.eval_fk(m, m.joint_q, m.joint_qd, s0)
v = ViewerGL(width=800, height=450, headless=True, vsync=False); v.set_model(m)
look = np.array([0.0,0.25,0.05]); eye = look + np.array([0.0,-0.7,0.5])
d = look-eye
v.set_camera(tuple(float(x) for x in eye),
             math.degrees(math.atan2(d[2], float(np.hypot(d[0],d[1])))),
             math.degrees(math.atan2(d[1], d[0])))
v.begin_frame(0.0); v.log_state(s0); v.end_frame()
f = v.get_frame(); a = f.numpy() if hasattr(f,"numpy") else np.asarray(f)
if a.dtype != np.uint8: a = (np.clip(a,0,1)*255).astype(np.uint8)
a = a[:,:,:3]
Image.fromarray(a).save("flip_raw.png")
Image.fromarray(np.flipud(a)).save("flip_flipped.png")
# Which half is darker? Camera looks DOWN at a bench, so the bench (bright)
# must occupy the LOWER half and the background (dark) the upper half.
top = a[: a.shape[0]//2].mean(); bot = a[a.shape[0]//2 :].mean()
print(f"raw     : top mean={top:6.1f}  bottom mean={bot:6.1f}")
print(f"correct orientation is: {'raw (no flip)' if top < bot else 'flipped (np.flipud)'}")
v.close()
