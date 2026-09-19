#!/usr/bin/env bash
cd /home/ujaan/isaac/labgen
D=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
for src in "$D/yam.xml" "$D/yam.urdf"; do
for cfg in "300 30 10" "2000 150 10" "2000 150 100"; do
  set -- $cfg
  echo "=== $(basename $src)  ke=$1 kd=$2 eff=$3 ==="
  $PY test_drive.py "$src" "$1" "$2" "$3" 2>/dev/null | grep -E "joint_target_mode|joint_effort|max joint|VERDICT"
done; done
