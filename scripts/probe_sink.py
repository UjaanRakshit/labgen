"""Sink vs wall thickness, base area and load, across every catalog vessel.

The reported result was "the petri dish fails, the others pass". That reads as
one object being special. But the beaker sinks 1.29 mm with a 1.5 mm floor, and
landing under a 2 mm gate by 0.2 mm is luck, not a pass. If thin-walled vessels
sink as a class then the gate is passing three objects it should not.

Measures every open vessel in the catalog, on its own, so objects cannot
perturb each other.
"""
import sys, math, json, numpy as np, newton
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.catalog import CATALOG
from labgen.types import SceneSpec
from labgen import usda

VESSELS = [k for k, i in CATALOG.items()
           if i.shape in ("open_vessel", "conical_vessel")]

def scene_for(key):
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
    import pathlib, tempfile
    p = pathlib.Path(tempfile.gettempdir()) / f"{key}.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    return usda.write_scene(SceneSpec.read(p), f"/tmp/solo_{key}.usda")

def settle(path, dt=1/240, secs=2.0, ke=None):
    b = newton.ModelBuilder(); r = b.add_usd(str(path))
    if ke is not None:
        for i in range(b.shape_count):
            b.shape_material_ke[i] = ke
            b.shape_material_kd[i] = ke / 25.0
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
    return float(s0.body_q.numpy()[idx, 2] - z0)

print(f"{'vessel':24} {'wall':>6} {'base_d':>7} {'area':>8} {'mass':>7} {'press':>7} {'sink':>8}")
print(f"{'':24} {'mm':>6} {'mm':>7} {'mm2':>8} {'g':>7} {'Pa':>7} {'mm':>8}")
rows = []
for key in VESSELS:
    it = CATALOG[key]
    d = it.dims
    wall = d["wall"] * 1000
    base_d = (d.get("base_d") or d.get("outer_d")) * 1000
    area = math.pi * (base_d / 2) ** 2
    press = it.mass_kg * 9.81 / (area * 1e-6)
    sink = settle(scene_for(key)) * 1000
    rows.append((key, wall, base_d, area, it.mass_kg * 1000, press, sink))
    flag = "  <-- over 2 mm" if abs(sink) > 2.0 else ""
    print(f"{key:24} {wall:6.1f} {base_d:7.1f} {area:8.0f} {it.mass_kg*1000:7.1f} "
          f"{press:7.0f} {sink:8.2f}{flag}")

a = np.array([[r[1], r[3], r[5], abs(r[6])] for r in rows if np.isfinite(r[6])])
if len(a) > 3:
    print("\ncorrelation of |sink| with:")
    for name, col in (("wall thickness", 0), ("base area", 1), ("contact pressure", 2)):
        print(f"   {name:18} r = {np.corrcoef(a[:, col], a[:, 3])[0,1]:+.3f}")
