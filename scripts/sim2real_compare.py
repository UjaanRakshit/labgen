"""Compare two sim-to-real logs of the same protocol.

    python scripts/sim2real_compare.py real_can0.npz newton_sim.npz [--tol 0.5]

Prints per-joint RMS and max difference, each side's own tracking error, and the
step-response figures side by side. Exits 0 when the two agree within the
tolerance, 1 when they do not -- so it can gate a CI job once real logs exist.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from labgen.sim2real import Log, compare                                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--tol", type=float, default=0.5, help="RMS tolerance, degrees")
    args = ap.parse_args()
    c = compare(Log.load(args.a), Log.load(args.b), tolerance_deg=args.tol)
    print(c)
    return 0 if c.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
