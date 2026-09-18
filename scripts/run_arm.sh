#!/usr/bin/env bash
set -uo pipefail
cd /home/ujaan/isaac/labgen
Y=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
rm -rf frames_arm
timeout 900 /home/ujaan/isaac/IsaacLab/.venv/bin/python bench_with_arm.py bench_5.usda "$Y" frames_arm --seconds 2
