#!/usr/bin/env bash
cd /home/ujaan/isaac/labgen
D=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
for src in "$D/yam.urdf" "$D/yam.xml"; do
  echo "=== $(basename $src) ==="
  $PY cah.py "$src" 400 40 10 2>/dev/null | grep -E "dof=|effort limits|sag from|max \||VERDICT"
done
