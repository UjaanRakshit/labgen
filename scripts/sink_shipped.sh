#!/usr/bin/env bash
# The solo-vessel table at the constants labgen.settle ACTUALLY ships, plus the
# real settle test on both reference scenes, so the two cannot disagree without
# it being visible.
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
cd /home/ujaan/isaac/labgen || exit 1
{
  echo "######## solo vessels at the SHIPPED constants (ke=160000, kd=800, dt=1/960) ########"
  "$PY" probe_sink.py 160000 960 800 2>/dev/null
  echo
  echo "######## the real settle test, both reference scenes ########"
  "$PY" - <<'PYEOF' 2>/dev/null
import sys
sys.path.insert(0, "/mnt/c/Ujaan Docx/Research/labgen")
from labgen.settle import settle, NewtonBackend, CONTACT_KE, CONTACT_KD, SETTLE_DT
print(f"labgen.settle ships ke={CONTACT_KE:g} kd={CONTACT_KD:g} dt=1/{1/SETTLE_DT:.0f}")
for scene in ("bench_5", "bench_arm"):
    print()
    print(settle(f"/home/ujaan/isaac/labgen/{scene}.usda", backend=NewtonBackend()))
PYEOF
} | tee /tmp/sink_shipped.log
