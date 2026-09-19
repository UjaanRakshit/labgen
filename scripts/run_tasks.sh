#!/usr/bin/env bash
set -uo pipefail
cd /home/ujaan/isaac/labgen
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
ARM=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
rm -rf frames_tasks
timeout 3600 $PY demo_tasks.py bench_arm.usda "$ARM" frames_tasks
