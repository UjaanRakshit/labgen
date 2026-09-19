#!/usr/bin/env bash
U=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1/yam.urdf
echo "collision tags in URDF: $(grep -c '<collision' "$U")"
echo "visual tags in URDF   : $(grep -c '<visual' "$U")"
echo "--- a link ---"
sed -n '/<link name="link1"/,/<\/link>/p' "$U" | head -30
