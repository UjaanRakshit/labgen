"""Does convex_decomposition keep the petri dish's cavity open?

Fixing the settle test by sealing a vessel would be the exact failure CLAUDE.md
hard rule 1 exists to prevent -- a policy could then "place" something in the
dish by resting it on a lid that is not there. Convex DECOMPOSITION should be
safe where a single hull is not, because the pieces follow the concavity. Should
be is not is, so: drop a test tube into the dish and see where it stops.
"""
import sys, dataclasses, numpy as np, newton, json
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
from labgen import usda

scene = {
  "schema_version": 1, "name": "petri_cavity", "robot_base_frame": "yam_base_link",
  "source": {"method":"hand_authored","marker_config":"not_applicable_hand_authored",
             "video":"none","git_sha":"0"*40,
             "note":"Drop a test tube into a petri dish. Dish floor is z=0.0012, rim is z=0.015."},
  "objects": [
    {"instance_id":"bench","catalog_key":"bench_top","position":[0,0.3,-0.04],
     "orientation_wxyz":[1,0,0,0],"fixed":True,"confidence":1.0,"provenance":"manual",
     "generated":None,"note":""},
    {"instance_id":"petri","catalog_key":"petri_dish_100","position":[0,0.3,0],
     "orientation_wxyz":[1,0,0,0],"fixed":False,"confidence":1.0,"provenance":"manual",
     "generated":None,"note":""},
    {"instance_id":"probe","catalog_key":"test_tube_16x100","position":[0,0.3,0.12],
     "orientation_wxyz":[0.7071068,0,0.7071068,0],"fixed":False,"confidence":1.0,
     "provenance":"manual","generated":None,"note":"laid horizontally, dropped from 120mm"},
  ]}
import pathlib
pathlib.Path("/tmp/pc.json").write_text(json.dumps(scene), encoding="utf-8")

def run(collider):
    orig = CATALOG["petri_dish_100"]
    CATALOG["petri_dish_100"] = dataclasses.replace(orig, collider=collider)
    try:
        p = usda.write_scene(SceneSpec.read("/tmp/pc.json"), f"/tmp/pc_{collider}.usda").as_posix()
    finally:
        CATALOG["petri_dish_100"] = orig
    b = newton.ModelBuilder(); r = b.add_usd(p); m = b.finalize()
    paths = {v:k for k,v in r["path_body_map"].items()}
    idx = {paths[i].replace("/World/",""): i for i in range(m.body_count)}
    s0,s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    for _ in range(int(2.5*240)):
        ct = m.collide(s0); solver.step(s0,s1,c,ct,1/240); s0,s1=s1,s0
        if not np.isfinite(s0.body_q.numpy()).all(): print(f"{collider}: DIVERGED"); return
    z = s0.body_q.numpy()[idx["probe"], 2]
    verdict = ("INSIDE the dish" if z < 0.013 else
               "ON TOP -- the dish is SEALED" if z > 0.020 else "at the rim, ambiguous")
    print(f"  {collider:22} probe settles at z={z*1000:6.2f} mm  ->  {verdict}")

print("petri floor z = 1.2 mm, rim z = 15.0 mm; an 8 mm-radius tube lying in it rests near z=9.2 mm")
run("sdf")
run("convex_decomposition")
run("convex_hull")
