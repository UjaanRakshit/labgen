"""Which collider does SolverMuJoCo actually handle well for thin vessels?

MuJoCo's native narrowphase is convex-convex. Handing it a raw non-convex
triangle shell (approximation="none") is its worst case, which is exactly what
the petri dish is: 100 mm wide, 15 mm tall, a 1.2 mm floor. It sinks 8 mm --
more than its own wall thickness, i.e. it is passing through the worktop.

convex_decomposition would give MuJoCo convex pieces while still preserving the
cavity, so it does not violate hard rule 1 the way a single hull would.
"""
import sys, numpy as np, newton, dataclasses
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
from labgen import usda

SCENE = "/mnt/c/Ujaan Docx/Research/labgen/examples/bench_arm.json"

def settle(path, label):
    b = newton.ModelBuilder(); r = b.add_usd(path)
    m = b.finalize()
    paths = {v:k for k,v in r["path_body_map"].items()}
    s0,s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    q0 = s0.body_q.numpy().copy()
    for _ in range(480):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1=s1,s0
        if not np.isfinite(s0.body_q.numpy()).all(): print(f"{label}: DIVERGED"); return
    q1 = s0.body_q.numpy()
    print(f"{label:26} " + "  ".join(
        f"{paths.get(i,'?').replace('/World/',''):9}={(q1[i,2]-q0[i,2])*1000:+7.2f}mm"
        for i in range(m.body_count)))

orig = {k: CATALOG[k] for k in ("petri_dish_100","beaker_250","test_tube_16x100")}
try:
    settle("/tmp/a.usda" if False else usda.write_scene(SceneSpec.read(SCENE), "/tmp/base.usda").as_posix(), "sdf/none (current)")
    for coll in ("convex_decomposition",):
        for k in orig:
            CATALOG[k] = dataclasses.replace(orig[k], collider=coll)
        p = usda.write_scene(SceneSpec.read(SCENE), f"/tmp/{coll}.usda").as_posix()
        settle(p, coll)
    # decomposition on the petri only
    for k in orig: CATALOG[k] = orig[k]
    CATALOG["petri_dish_100"] = dataclasses.replace(orig["petri_dish_100"],
                                                    collider="convex_decomposition")
    p = usda.write_scene(SceneSpec.read(SCENE), "/tmp/petri_only.usda").as_posix()
    settle(p, "petri decomp only")
finally:
    for k,v in orig.items(): CATALOG[k] = v
