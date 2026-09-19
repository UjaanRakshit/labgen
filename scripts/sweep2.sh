#!/usr/bin/env bash
cd /home/ujaan/isaac/labgen
D=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
for src in "$D/yam.xml" "$D/yam.urdf"; do
for cfg in "100 10 10" "400 40 10" "1000 50 10"; do
  set -- $cfg
  echo "=== $(basename $src) ke=$1 kd=$2 eff=$3 ==="
  $PY cah.py "$src" "$1" "$2" "$3" 2>/dev/null | grep -E "sag from|max \||VERDICT"
done; done
