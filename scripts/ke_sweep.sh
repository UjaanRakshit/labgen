#!/usr/bin/env bash
# The bind: a position PD with no gravity feedforward has no gain that works.
#
# Low ke cannot generate the 7.26 N.m gravity load without a large position
# error, so the arm sags. High ke generates it from a tiny error, so it holds --
# but then it saturates the 10 N.m limit at effort/ke of lag (0.19 deg at
# ke=3000) and goes open-loop the moment the arm actually moves.
#
# Every run below is at the rated 10 N.m. If no gain tracks the trajectory, the
# fix is gravity compensation, not a bigger motor.
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
LOG=/tmp/ke_sweep.log
cd /home/ujaan/isaac/labgen || exit 1
URDF=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
: > "$LOG"
for ke in "$@"; do
    echo "================ ke = ${ke} at the rated 10 N.m ================" | tee -a "$LOG"
    LABGEN_KE="$ke" "$PY" demo_tasks.py bench_arm.usda "$URDF" /tmp/ke_out --no-render 2>/dev/null \
        | grep -E 'j1=|worst |peak arm|moved ' | tee -a "$LOG"
done
