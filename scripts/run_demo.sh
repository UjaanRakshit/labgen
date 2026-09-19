#!/usr/bin/env bash
set -uo pipefail
cd /home/ujaan/isaac/labgen
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
ARM=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.xml
rm -rf frames_demo
timeout 1800 $PY demo_reach.py bench_5.usda "$ARM" frames_demo
