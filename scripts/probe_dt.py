import sys, numpy as np, newton, dataclasses
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
from labgen import usda
SCENE = "/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json"

def settle(path, dt, label, secs=2.0):
    b = newton.ModelBuilder(); r = b.add_usd(path); m = b.finalize()
    paths = {v:k for k,v in r["path_body_map"].items()}
    s0,s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    q0 = s0.body_q.numpy().copy()
    for _ in range(int(secs/dt)):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,dt); s0,s1=s1,s0
        if not np.isfinite(s0.body_q.numpy()).all(): print(f"{label}: DIVERGED"); return
    q1 = s0.body_q.numpy()
    worst = max(abs(q1[i,2]-q0[i,2]) for i in range(m.body_count))*1000
    print(f"{label:28} worst={worst:6.2f}mm  " + " ".join(
        f"{paths.get(i,'?').replace('/World/','')[:9]}={(q1[i,2]-q0[i,2])*1000:+6.2f}"
        for i in range(m.body_count)))

orig = CATALOG["petri_dish_100"]
CATALOG["petri_dish_100"] = dataclasses.replace(orig, collider="convex_decomposition")
p = usda.write_scene(SceneSpec.read(SCENE), "/tmp/pd.usda").as_posix()
CATALOG["petri_dish_100"] = orig
for dt, nm in ((1/240,"dt=1/240"), (1/480,"dt=1/480"), (1/960,"dt=1/960")):
    settle(p, dt, f"petri-decomp {nm}")
