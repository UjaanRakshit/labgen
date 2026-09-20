"""Is the sink a contact ALLOWANCE rather than compliance?

All five beakers sink ~1.25 mm across a 13x mass range, and sink correlates
negatively with contact pressure. Neither is what a spring-damper contact does.
A fixed allowance -- MuJoCo's margin/gap, where contacts are detected early and
only become active past a threshold -- would look exactly like this.
"""
import sys, json, pathlib, numpy as np, newton
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.types import SceneSpec
from labgen import usda as U

OUT = pathlib.Path("/home/ujaan/isaac/labgen/solo")
OUT.mkdir(exist_ok=True)

def scene_for(key):
    """Regenerate here rather than reuse /tmp -- WSL wipes it between sessions."""
    d = {"schema_version": 1, "name": f"solo_{key}", "robot_base_frame": "yam_base_link",
         "source": {"method": "hand_authored", "marker_config": "not_applicable_hand_authored",
                    "video": "none", "git_sha": "0"*40, "note": "single-vessel settle probe"},
         "objects": [
             {"instance_id": "bench", "catalog_key": "bench_top", "position": [0, 0.3, -0.04],
              "orientation_wxyz": [1,0,0,0], "fixed": True, "confidence": 1.0,
              "provenance": "manual", "generated": None, "note": ""},
             {"instance_id": "obj", "catalog_key": key, "position": [0, 0.3, 0.0],
              "orientation_wxyz": [1,0,0,0], "fixed": False, "confidence": 1.0,
              "provenance": "manual", "generated": None, "note": ""}]}
    j = OUT / f"{key}.json"; j.write_text(json.dumps(d), encoding="utf-8")
    return str(U.write_scene(SceneSpec.read(j), OUT / f"{key}.usda"))

def build(usda, **over):
    b = newton.ModelBuilder(); r = b.add_usd(usda)
    for i in range(b.shape_count):
        for k, v in over.items():
            getattr(b, f"shape_{k}")[i] = v
    m = b.finalize()
    return m, {v: k for k, v in r["path_body_map"].items()}

P, B, T = (scene_for("petri_dish_100"), scene_for("beaker_250"),
           scene_for("test_tube_16x100"))
m, paths = build(P)
print("as imported:")
for nm in ("gap", "margin", "material_ke", "material_kd", "material_kf", "thickness"):
    a = getattr(m, f"shape_{nm}", None)
    if a is not None:
        print(f"   shape_{nm:14} {np.unique(a.numpy())[:4]}")

def settle(usda, dt=1/240, secs=2.0, **over):
    m, paths = build(usda, **over)
    idx = [i for i in range(m.body_count) if paths.get(i, "").endswith("obj")][0]
    s0, s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    z0 = s0.body_q.numpy()[idx, 2]
    for _ in range(int(secs / dt)):
        ct = m.collide(s0); solver.step(s0, s1, c, ct, dt); s0, s1 = s1, s0
        if not np.isfinite(s0.body_q.numpy()).all():
            return float("nan")
    return float(s0.body_q.numpy()[idx, 2] - z0) * 1000

print(f"\n{'override':34} {'petri':>9} {'beaker':>9} {'tube':>9}  (mm)")
for label, kw in (
    ("as imported", {}),
    ("gap=0.0", {"gap": 0.0}),
    ("gap=0.0 margin=0.0", {"gap": 0.0, "margin": 0.0}),
    ("gap=0.001", {"gap": 0.001}),
    ("gap=0.0 ke=1e4", {"gap": 0.0, "material_ke": 1e4, "material_kd": 400.0}),
    ("gap=0.0 ke=5e4", {"gap": 0.0, "material_ke": 5e4, "material_kd": 2000.0}),
):
    vals = [settle(p, **kw) for p in (P, B, T)]
    print(f"{label:34} " + " ".join(f"{v:9.2f}" for v in vals))
