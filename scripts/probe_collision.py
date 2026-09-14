"""Is the SDF collider real, or did Newton quietly fall back to something else?

`shape_type == MESH` for all five objects says only that the geometry is a
triangle mesh. It says nothing about how collisions against that mesh are
resolved, and a silent fallback to a convex hull is exactly the failure this
project treats as unacceptable: a hulled beaker is sealed, and a policy scores
a success by resting an object on a lid that does not exist.

So: enumerate what Newton actually recorded per shape, and print it. No
assertions here -- this is a probe, and its job is to show what is there.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

USDA = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_5.usda")

import newton  # noqa: E402


def hdr(t: str) -> None:
    print(f"\n{'=' * 4} {t} {'=' * max(0, 66 - len(t))}")


builder = newton.ModelBuilder()
result = builder.add_usd(str(USDA))
model = builder.finalize()

shape_paths = {v: k for k, v in result["path_shape_map"].items()}

hdr("per-shape properties Newton recorded")
interesting = [a for a in dir(model) if a.startswith("shape_")]
print(f"   available: {interesting}")

hdr("what each shape got")


def arr(name):
    a = getattr(model, name, None)
    if a is None:
        return None
    try:
        return a.numpy()
    except Exception:
        return a


for i in range(model.shape_count):
    path = shape_paths.get(i, "?")
    print(f"\n   [{i}] {path}")
    for name in interesting:
        a = arr(name)
        if a is None or not hasattr(a, "__len__") or len(a) <= i:
            continue
        v = a[i]
        if isinstance(v, np.ndarray) and v.size > 8:
            print(f"        {name:26} array{v.shape}")
        else:
            print(f"        {name:26} {v}")

hdr("shape_source: the geometry object behind each shape")
src = getattr(model, "shape_source", None)
if src is not None:
    for i in range(model.shape_count):
        g = src[i] if i < len(src) else None
        path = shape_paths.get(i, "?")
        print(f"   [{i}] {path:28} {type(g).__name__}")
        if g is None:
            continue
        for attr in ("has_sdf", "sdf", "volume", "vertices", "indices", "is_solid",
                     "maxhullvert", "remesh", "sdf_resolution"):
            if hasattr(g, attr):
                val = getattr(g, attr)
                if isinstance(val, np.ndarray):
                    print(f"        {attr:20} array{val.shape}")
                else:
                    print(f"        {attr:20} {val}")

hdr("newton Mesh class: what SDF support exists at all")
print(f"   newton.Mesh attrs: {[a for a in dir(newton.Mesh) if not a.startswith('__')]}")
for name in ("SDF", "Sdf", "sdf"):
    if hasattr(newton, name):
        print(f"   newton.{name} exists: {getattr(newton, name)}")

hdr("solvers available")
import newton.solvers as S  # noqa: E402
print(f"   {[a for a in dir(S) if a.startswith('Solver')]}")
