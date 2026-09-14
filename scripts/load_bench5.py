"""Load bench_5.usda for real: Isaac's own USD runtime, then Newton physics.

Run with Isaac Lab's venv:
    ~/isaac/IsaacLab/.venv/bin/python scripts/load_bench5.py <path-to-bench_5.usda>

Three questions, in increasing order of what they cost to answer:

1. Does the stage parse in the USD version Isaac actually ships (usd-exchange
   2.3.0 vendors USD 25.5), not just in the usd-core 26.8 wheel labgen tests
   against?
2. Does Newton import it, and does it keep our collider choices?
3. Does SDF collision geometry actually get built for a 1 mm wall
   (test_tube_16x100) and a 100 x 15 mm dish (petri_dish_100)? Those two were
   put in bench_5 specifically to stress it.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

USDA = Path(sys.argv[1] if len(sys.argv) > 1 else "out/bench_5.usda")

EXPECTED = {
    "bench": ("boundingCube", True),
    "hotplate": ("boundingCube", False),
    "beaker": ("sdf", False),
    "test_tube": ("sdf", False),
    "petri": ("sdf", False),
}


def hdr(title: str) -> None:
    print(f"\n{'=' * 4} {title} {'=' * (68 - len(title))}")


def q1_usd_parses() -> bool:
    hdr("1. parse in Isaac's USD runtime")
    from pxr import Tf, Usd, UsdGeom, UsdPhysics

    print(f"   USD version: {Usd.GetVersion()}")
    print(f"   file: {USDA}  ({USDA.stat().st_size / 1024:.0f} KB)")

    mark = Tf.Error.Mark()
    mark.SetMark()
    stage = Usd.Stage.Open(str(USDA))
    errors = [str(e.commentary).strip() for e in mark.GetErrors()]

    if errors:
        print(f"   FAIL: {len(errors)} USD diagnostic(s):")
        for e in errors[:8]:
            print(f"     - {e}")
        return False
    if stage is None:
        print("   FAIL: stage is None")
        return False

    print("   stage opened with zero diagnostics")
    print(f"   metersPerUnit={UsdGeom.GetStageMetersPerUnit(stage)} "
          f"upAxis={UsdGeom.GetStageUpAxis(stage)}")

    ok = True
    for name, (approximation, fixed) in EXPECTED.items():
        prim = stage.GetPrimAtPath(f"/World/{name}")
        if not prim.IsValid():
            print(f"   FAIL: /World/{name} missing")
            ok = False
            continue
        got = UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()
        rb = bool(UsdPhysics.RigidBodyAPI(prim))
        flag = "ok " if (got == approximation and rb == (not fixed)) else "BAD"
        print(f"   {flag} {name:11} approximation={got!r:22} rigidBody={rb}")
        if flag == "BAD":
            ok = False
    return ok


def q2_newton_api():
    hdr("2. Newton USD import API")
    import newton

    print(f"   newton {newton.__version__}")
    builder_attrs = [a for a in dir(newton.ModelBuilder)
                     if "usd" in a.lower() or "add_" in a.lower()]
    print(f"   ModelBuilder import entry points: {builder_attrs[:12]}")

    importers = []
    for modname in ("newton.utils", "newton._src.utils", "newton.importers", "newton.usd"):
        try:
            mod = __import__(modname, fromlist=["*"])
            hits = [a for a in dir(mod) if "usd" in a.lower()]
            if hits:
                importers.append((modname, hits))
        except Exception:
            pass
    for modname, hits in importers:
        print(f"   {modname}: {hits[:8]}")
    return newton


def q3_newton_import(newton):
    hdr("3. Newton import + collision shapes")
    builder = newton.ModelBuilder()

    fn = getattr(builder, "add_usd", None)
    if fn is None:
        print("   no ModelBuilder.add_usd; trying newton.utils.parse_usd")
        from newton.utils import parse_usd  # type: ignore
        result = parse_usd(str(USDA), builder)
    else:
        result = fn(str(USDA))

    print(f"   import returned: {type(result).__name__}")
    if isinstance(result, dict):
        for k in sorted(result):
            v = result[k]
            print(f"     {k}: {type(v).__name__} "
                  f"{len(v) if hasattr(v, '__len__') else v}")

    print(f"   builder shapes: {builder.shape_count}")
    print(f"   builder bodies: {builder.body_count}")

    model = builder.finalize()
    print(f"   finalized model: {model.shape_count} shapes, {model.body_count} bodies")

    # What geometry type did each shape get?
    import numpy as np
    types = model.shape_type.numpy() if hasattr(model.shape_type, "numpy") else None
    if types is not None:
        names = {}
        for attr in dir(newton.GeoType if hasattr(newton, "GeoType") else object):
            if attr.isupper():
                names[getattr(newton.GeoType, attr)] = attr
        from collections import Counter
        print(f"   shape geo types: "
              f"{ {names.get(int(t), int(t)): n for t, n in Counter(types.tolist()).items()} }")
    return model


def main() -> int:
    if not USDA.exists():
        print(f"no such file: {USDA}")
        return 2

    if not q1_usd_parses():
        print("\nSTOP: the stage does not parse in Isaac's USD runtime.")
        return 1

    try:
        newton = q2_newton_api()
    except Exception:
        traceback.print_exc()
        return 1

    try:
        q3_newton_import(newton)
    except Exception:
        print("   Newton import FAILED:")
        traceback.print_exc()
        return 1

    print("\nAll three questions answered.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
