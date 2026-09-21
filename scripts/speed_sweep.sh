#!/usr/bin/env bash
# Does the task fail at the rated 10 N.m because the arm is incapable, or
# because the trajectory asks for acceleration torque on top of gravity?
#
# Same scene, same effort limit, same gains, no rendering. Only the waypoint
# durations move. If peak tracking error falls with SPEED_SCALE, the trajectory
# was the problem and the 10 N.m rating is not the obstacle.
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
LOG=/tmp/speed_sweep.log
cd /home/ujaan/isaac/labgen || exit 1
URDF=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
: > "$LOG"
for s in "$@"; do
    echo "================ SPEED_SCALE = ${s}x ================" | tee -a "$LOG"
    LABGEN_SPEED_SCALE="$s" "$PY" demo_tasks.py bench_arm.usda "$URDF" /tmp/sweep_out \
        --no-render 2>/dev/null \
        | grep -E 'speed scale|tracking error|attach|released|placed|moved |peak arm' \
        | tee -a "$LOG"
done
echo "--- log at $LOG ---"
