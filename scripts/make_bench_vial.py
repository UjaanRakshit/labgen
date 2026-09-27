"""Author examples/bench_vial.json: bench_arm plus a 20 mL scintillation vial.

For the vial task -- pick the vial, set it on the hotplate, take it back off.
bench_arm is left untouched (the grasp acceptance and the baseline task measure
against it); this adds one object and nothing else moves.

The vial is vial_20ml: OD 28 mm x 61 mm, confirmed; its neck and shoulder are
NOT in the catalog (TODO), so the task grasps the straight BODY at mid-height,
where the geometry is known, never the neck.

Placement is checked by labgen.validate, not eyeballed.

    .venv/Scripts/python scripts/make_bench_vial.py
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from labgen import usda                      # noqa: E402
from labgen.types import SceneSpec           # noqa: E402
from labgen.validate import validate         # noqa: E402

# Same convention as make_bench_arm: angle from +y, radius from the robot base.
# -14 deg, 0.31 m: inside the YAM's reach, 0.13 m from the petri dish and
# 0.16 m from the test tube, so a 94 mm open jaw clears both.
VIAL_DEG, VIAL_R = -14.0, 0.31

src = json.loads((ROOT / "examples/bench_arm.json").read_text(encoding="utf-8"))
a = math.radians(VIAL_DEG)
src["objects"].append({
    "instance_id": "vial",
    "catalog_key": "vial_20ml",
    "position": [round(VIAL_R * math.sin(a), 4), round(VIAL_R * math.cos(a), 4), 0.0],
    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    "fixed": False,
    "confidence": 1.0,
    "provenance": "manual",
    "generated": None,
    "note": (f"{VIAL_DEG:+.0f} deg, {VIAL_R:.2f} m from the robot base. Grasped at "
             f"the body; the catalog has no neck geometry (TODO)."),
})
src["name"] = "bench_vial"
src["source"]["note"] = ("bench_arm plus a 20 mL scintillation vial, for the vial-on-"
                         "hotplate task. Every other object is where bench_arm has it.")
src["source"]["git_sha"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                          capture_output=True, text=True).stdout.strip()

out = ROOT / "examples/bench_vial.json"
out.write_text(json.dumps(src, indent=2) + "\n", encoding="utf-8")
spec = SceneSpec.read(out)
report = validate(spec)
print(report.format() if hasattr(report, "format") else report)
usd = usda.write_scene(spec, ROOT / "out/bench_vial.usda")
print(f"wrote {out}\nwrote {usd}")
