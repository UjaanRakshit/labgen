#!/usr/bin/env bash
# Phone teleop of the Isaac Lab bench task, recording demos. Runs in WSL.
# Start the phone page on Windows first:  .venv/Scripts/python scripts/teleop_touch.py
#
#   run_teleop_task.sh [out.hdf5] [extra teleop_task.py args...]
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
bash "/mnt/c/Ujaan Docx/Research/labgen/scripts/sync.sh" >/dev/null
cd /home/ujaan/isaac/labgen || exit 1
OUT="${1:-/home/ujaan/isaac/labgen/demos/bench_$(date +%Y%m%d_%H%M%S).hdf5}"
shift || true
# Anchored: an unanchored pattern also matches the bash -c that runs this script.
pkill -9 -f '^/home/ujaan/isaac/IsaacLab/.venv/bin/python teleop_(task|sim)' 2>/dev/null
exec "$PY" teleop_task.py --stream --out "$OUT" "$@" 2>&1 \
  | grep --line-buffered -vE '^Module|^Warp|linesearch|root com velocity|^\||^\+|\[INFO\]'
