#!/usr/bin/env bash
cd /home/ujaan/isaac/labgen
Y=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
for cfg in "400 40 10" "2000 100 10" "2000 100 200" "8000 400 200"; do
  set -- $cfg
  echo "=== ke=$1 kd=$2 effort=$3 ==="
  $PY cah.py "$Y" "$1" "$2" "$3" 2>/dev/null | grep -E "effort limits|target_ke in|max \||VERDICT|sag from"
done
