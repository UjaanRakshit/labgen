#!/usr/bin/env bash
# Start the teleop sim. Exists because quoting a grep filter through
# `wsl -- bash -c "..."` from the Windows side mangles the command line; every
# multi-step WSL action in this repo lives in a file for the same reason.
#
#   run_teleop.sh [seconds] [--bimanual]
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
cd /home/ujaan/isaac/labgen || exit 1
URDF=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
SECS="${1:-7200}"
shift || true
# Anchored to the start of the python process's own command line. An
# unanchored `pkill -f teleop_sim` also matches the `bash -c` that runs
# this script, and kills its own shell before anything happens.
pkill -9 -f '^/home/ujaan/isaac/IsaacLab/.venv/bin/python teleop_sim' 2>/dev/null
sleep 1
exec "$PY" teleop_sim.py bench_arm.usda "$URDF" --stream --seconds "$SECS" "$@" 2>&1 \
  | grep --line-buffered -vE '^Module|^Warp|^   (CUDA|Devices|Kernel)|^     "|^     /home|linesearch|Deprecation|contacts ='
