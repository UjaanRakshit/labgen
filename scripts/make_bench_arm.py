"""Author examples/bench_arm.json: bench_5's objects, placed for the arm.

bench_5 was laid out before there was a robot in the scene, so its objects sit
inside the arm's swept volume -- the reach knocks the beaker over. That scene
stays as it is: it is the physics reference the settle test and the USD tests
run against, and moving things to suit a demo would quietly invalidate every
number measured against it.

This is a second scene instead, with the same catalog items arranged on an arc
at a comfortable reach. Positions are computed and checked here rather than
eyeballed.
"""
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"c:\Ujaan Docx\Research\labgen")
sys.path.insert(0, str(ROOT))

from labgen.catalog import CATALOG          # noqa: E402
from labgen import meshes                    # noqa: E402
from labgen.types import SceneSpec           # noqa: E402

sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                     capture_output=True, text=True).stdout.strip()

# The YAM's sampled workspace reaches 0.74 m from the base, and its home pose
# puts the gripper at 0.21 m. An arc at 0.40-0.46 m is comfortably inside that
# and comfortably outside the arm's own body at home.
LAYOUT = [
    # instance,    catalog key,          angle deg from +y,  radius m
    ("hotplate",   "hotplate_stirrer",   -52.0, 0.46),
    ("petri",      "petri_dish_100",     -24.0, 0.43),
    ("test_tube",  "test_tube_16x100",     4.0, 0.42),
    ("beaker",     "beaker_250",          30.0, 0.40),
]

# The robot base is the origin, and the YAM's base plate plus its elbow reach
# roughly 0.25 m BEHIND it at the home pose (link3 sits at x = -0.244). Centre
# the worktop far enough forward that the whole footprint is supported --
# with 0.36 the base plate overhung the near edge and the arm appeared to be
# bolted to thin air.
BENCH_CENTRE_Y = 0.26

objects = [{
    "instance_id": "bench",
    "catalog_key": "bench_top",
    "position": [0.0, BENCH_CENTRE_Y, -CATALOG["bench_top"].dims["z"]],
    "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
    "fixed": True,
    "confidence": 1.0,
    "provenance": "manual",
    "generated": None,
    "note": ("Static fixture, sunk by its own thickness so the worktop surface "
             "is exactly z=0 -- the plane the arm bolts to and the plane every "
             "object rests on."),
}]

for name, key, deg, radius in LAYOUT:
    rad = math.radians(deg)
    x = radius * math.sin(rad)
    y = radius * math.cos(rad)
    objects.append({
        "instance_id": name,
        "catalog_key": key,
        "position": [round(x, 4), round(y, 4), 0.0],
        "orientation_wxyz": [1.0, 0.0, 0.0, 0.0],
        "fixed": False,
        "confidence": 1.0,
        "provenance": "manual",
        "generated": None,
        "note": f"{deg:+.0f} deg, {radius:.2f} m from the robot base.",
    })

scene = {
    "schema_version": 1,
    "name": "bench_arm",
    "robot_base_frame": "yam_base_link",
    "source": {
        "method": "hand_authored",
        "note": ("Same catalog items as bench_5, re-arranged on an arc inside "
                 "the YAM's reachable workspace. bench_5 predates the arm and "
                 "puts objects inside its swept volume; it is left alone "
                 "because it is the reference the physics tests measure "
                 "against."),
        "marker_config": "not_applicable_hand_authored",
        "video": "none",
        "git_sha": sha,
        "frame_convention": ("Robot base at the origin, worktop surface at z=0, "
                             "bench extending in +y in front of the robot."),
    },
    "objects": objects,
}

path = ROOT / "examples" / "bench_arm.json"
path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
print(f"wrote {path}")

# --- check it before trusting it -----------------------------------------
import itertools

import numpy as np

spec = SceneSpec.read(path)
boxes = {}
for o in spec.objects:
    m = meshes.build(o.catalog_item())
    lo, hi = m.bounds()
    p = np.array(o.position)
    boxes[o.instance_id] = (lo + p, hi + p)

blo, bhi = boxes["bench"]
print(f"\nworktop surface z = {bhi[2]:+.4f}")
ok = True
for o in spec.free_bodies:
    lo, hi = boxes[o.instance_id]
    rest = abs(lo[2] - bhi[2]) < 1e-9
    on = lo[0] >= blo[0] and hi[0] <= bhi[0] and lo[1] >= blo[1] and hi[1] <= bhi[1]
    d = math.hypot(*o.position[:2])
    print(f"  {o.instance_id:10} reach {d:.3f} m  rests={rest}  on_bench={on}")
    ok &= rest and on

for a, b in itertools.combinations([o.instance_id for o in spec.free_bodies], 2):
    alo, ahi = boxes[a]
    blo2, bhi2 = boxes[b]
    if all(ahi[i] > blo2[i] and bhi2[i] > alo[i] for i in range(3)):
        print(f"  OVERLAP {a} / {b}")
        ok = False

# The arm at home occupies roughly a 0.25 m sphere around the base; nothing
# should be inside that.
for o in spec.free_bodies:
    lo, hi = boxes[o.instance_id]
    nearest = math.hypot(max(0.0, max(blo[0] - hi[0], lo[0] - bhi[0], 0.0)) or min(abs(lo[0]), abs(hi[0])) if lo[0] * hi[0] > 0 else 0.0,
                         min(abs(lo[1]), abs(hi[1])) if lo[1] * hi[1] > 0 else 0.0)
    if nearest < 0.25:
        print(f"  WARNING {o.instance_id} comes within {nearest:.3f} m of the arm base")

print("\nlayout OK" if ok else "\nLAYOUT PROBLEM")
