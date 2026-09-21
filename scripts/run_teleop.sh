#!/usr/bin/env bash
# Start the teleop sim. Exists because quoting a grep filter through
# `wsl -- bash -c "..."` from the Windows side mangles the command line; every
# multi-step WSL action in this repo lives in a file for the same reason.
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
cd /home/ujaan/isaac/labgen || exit 1
URDF=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
pkill -9 -f teleop_sim 2>/dev/null
sleep 1
exec "$PY" teleop_sim.py bench_arm.usda "$URDF" --stream --seconds "${1:-7200}" 2>&1 \
  | grep --line-buffered -vE '^Module|^Warp|^   (CUDA|Devices|Kernel)|^     |linesearch|Deprecation|contacts ='
