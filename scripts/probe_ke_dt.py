"""Contact stiffness against timestep, as a grid.

Earlier findings looked contradictory: raising ke helped then hurt, and
shrinking dt made things worse. Both were one-dimensional slices of a
two-dimensional problem. A penetration-based contact is stable only while the
stiffness is small relative to what the timestep can integrate, so ke and dt
have to move together -- holding one fixed and sweeping the other finds an
optimum that is an artifact of the fixed value.
"""
import sys, json, pathlib, numpy as np, newton
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.types import SceneSpec
from labgen import usda as U

OUT = pathlib.Path("/home/ujaan/isaac/labgen/solo"); OUT.mkdir(exist_ok=True)

def scene_for(key):
    d = {"schema_version": 1, "name": f"solo_{key}", "robot_base_frame": "yam_base_link",
         "source": {"method": "hand_authored", "marker_config": "not_applicable_hand_authored",
                    "video": "none", "git_sha": "0"*40, "note": "probe"},
         "objects": [
             {"instance_id": "bench", "catalog_key": "bench_top", "position": [0,0.3,-0.04],
              "orientation_wxyz": [1,0,0,0], "fixed": True, "confidence": 1.0,
              "provenance": "manual", "generated": None, "note": ""},
             {"instance_id": "obj", "catalog_key": key, "position": [0,0.3,0.0],
              "orientation_wxyz": [1,0,0,0], "fixed": False, "confidence": 1.0,
              "provenance": "manual", "generated": None, "note": ""}]}
    j = OUT / f"{key}.json"; j.write_text(json.dumps(d), encoding="utf-8")
    return str(U.write_scene(SceneSpec.read(j), OUT / f"{key}.usda"))

def settle(usda, ke, dt, secs=2.0):
    b = newton.ModelBuilder(); r = b.add_usd(usda)
    for i in range(b.shape_count):
        b.shape_material_ke[i] = ke
        b.shape_material_kd[i] = 2.0 * np.sqrt(ke)   # ~critical damping scaling
    m = b.finalize()
    paths = {v: k for k, v in r["path_body_map"].items()}
    idx = [i for i in range(m.body_count) if paths.get(i, "").endswith("obj")][0]
    s0, s1 = m.state(), m.state(); c = m.control()
    solver = newton.solvers.SolverMuJoCo(m, iterations=100, ls_iterations=50)
    z0 = s0.body_q.numpy()[idx, 2]
    for _ in range(int(secs / dt)):
        ct = m.collide(s0); solver.step(s0, s1, c, ct, dt); s0, s1 = s1, s0
        if not np.isfinite(s0.body_q.numpy()).all():
            return float("nan")
    return float(s0.body_q.numpy()[idx, 2] - z0) * 1000

P = scene_for("petri_dish_100")
KES = [2.5e3, 1e4, 4e4, 1.6e5]
DTS = [1/120, 1/240, 1/480, 1/960, 1/1920]
print("petri dish sink (mm), rows = contact stiffness, cols = timestep")
print(f"{'ke':>10} " + " ".join(f"{'1/'+str(int(1/d)):>9}" for d in DTS))
best = (None, 1e9)
for ke in KES:
    row = []
    for dt in DTS:
        v = settle(P, ke, dt)
        row.append(v)
        if np.isfinite(v) and abs(v) < best[1]:
            best = ((ke, dt), abs(v))
    print(f"{ke:10.0f} " + " ".join(f"{v:9.2f}" if np.isfinite(v) else f"{'diverged':>9}"
                                    for v in row))
(ke, dt), val = best
print(f"\nbest: ke={ke:.0f} dt=1/{1/dt:.0f} -> {val:.2f} mm")
print("\nsame setting applied to the other vessels:")
for key in ("beaker_250", "test_tube_16x100", "erlenmeyer_250", "graduated_cylinder_100"):
    print(f"   {key:24} {settle(scene_for(key), ke, dt):7.2f} mm")
