#!/usr/bin/env bash
Y=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1
echo "--- keyframe / home in MJCF ---"
grep -A3 -i "keyframe\|<key " "$Y/yam.xml" | head -20
echo "--- actuator / gains in MJCF ---"
grep -i "actuator\|position\|kp\|forcerange\|ctrlrange" "$Y/yam.xml" | head -20
echo "--- joint limits in URDF ---"
grep -o '<limit[^/]*/>' "$Y/yam.urdf" | head -10
echo "--- README home pose hints ---"
grep -i -B2 -A6 "home\|rest\|qpos\|zero position" "$Y/README.md" | head -40
