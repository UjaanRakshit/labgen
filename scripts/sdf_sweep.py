"""Why does the petri dish sink 8.4 mm?

Two candidate causes, and they imply different fixes:

  (a) SDF resolution -- the dish's bottom is under-resolved, so it penetrates
      until it finds material. Fix is per-object voxel sizing, and the
      constraint is catalog-wide: every thin-walled, wide object is affected.
  (b) Contact parameters -- stiffness/gap defaults let a light body settle into
      a surface regardless of collider fidelity. Fix is solver config, and the
      collider is innocent.

Distinguishing them is one experiment: sweep the authored voxel size and see
whether the sink depends on it. If (a), sink falls as voxels shrink. If (b),
sink is flat.

Usage: sdf_sweep.py <scene.usda-template-dir>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

import newton

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "bench_5.usda")
TARGET = "petri"
DURATION = 2.0
FPS = 240


def rewrite_voxel(text: str, prim: str, voxel: float | None) -> str:
    """Set (or strip) newton:sdfTargetVoxelSize on one object's geom."""
    start = text.index(f'def Xform "{prim}"')
    end = text.index('def Xform "', start + 10) if text.count('def Xform "', start + 10) else len(text)
    chunk = text[start:end]
    if voxel is None:
        chunk2 = re.sub(r"\n\s*float newton:sdfTargetVoxelSize = [^\n]*", "", chunk)
        chunk2 = chunk2.replace(', "NewtonSDFCollisionAPI"', "")
    else:
        chunk2 = re.sub(r"(float newton:sdfTargetVoxelSize = )[^\n]*",
                        rf"\g<1>{voxel:.9g}", chunk)
    return text[:start] + chunk2 + text[end:]


def settle(path: Path) -> tuple[dict[str, float], bool]:
    builder = newton.ModelBuilder()
    result = builder.add_usd(str(path))
    model = builder.finalize()
    paths = {v: k for k, v in result["path_body_map"].items()}

    s0, s1 = model.state(), model.state()
    control = model.control()
    start = s0.body_q.numpy().copy()
    solver = newton.solvers.SolverMuJoCo(model)

    dt = 1.0 / FPS
    for _ in range(int(DURATION * FPS)):
        contacts = model.collide(s0)
        solver.step(s0, s1, control, contacts, dt)
        s0, s1 = s1, s0
        if not np.isfinite(s0.body_q.numpy()).all():
            return {}, True

    end = s0.body_q.numpy()
    out = {}
    for i in range(model.body_count):
        name = paths.get(i, f"body{i}").replace("/World/", "")
        out[name] = float(end[i, 2] - start[i, 2])
    return out, False


def main() -> int:
    base = SRC.read_text(encoding="utf-8")
    tmp = SRC.parent / "_sweep.usda"

    print(f"sweeping newton:sdfTargetVoxelSize on '{TARGET}'")
    print(f"(petri_dish_100 is 100 x 100 x 15 mm with a 1.2 mm wall)\n")
    print(f"{'voxel':>12} {'grid (longest)':>16} {'petri dz':>12} {'beaker dz':>12} {'tube dz':>11}")
    print("-" * 68)

    for voxel in (None, 0.0016, 0.0008, 0.0004, 0.0002, 0.0001):
        tmp.write_text(rewrite_voxel(base, TARGET, voxel), encoding="utf-8")
        try:
            d, blew = settle(tmp)
        except Exception as e:
            print(f"{str(voxel):>12}  FAILED: {type(e).__name__}: {str(e)[:40]}")
            continue
        if blew:
            print(f"{str(voxel):>12}  DIVERGED")
            continue
        label = "no SDF API" if voxel is None else f"{voxel * 1000:.2f} mm"
        grid = "-" if voxel is None else f"{0.100 / voxel:.0f}"
        print(f"{label:>12} {grid:>16} {d.get('petri', float('nan')) * 1000:11.3f}mm "
              f"{d.get('beaker', float('nan')) * 1000:11.3f}mm "
              f"{d.get('test_tube', float('nan')) * 1000:10.3f}mm")

    tmp.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
