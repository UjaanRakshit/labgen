#!/usr/bin/env bash
# Copy the i2rt YAM + linear_4310 models from the WSL clone into a Windows dir
# so they can be diffed against the version the REAL robot runs.
set -u
SRC=/home/ujaan/isaac/i2rt/i2rt/robot_models
DST="$1"
mkdir -p "$DST"
for f in arm/yam/v1/yam.urdf arm/yam/v1/yam.xml gripper/linear_4310/linear_4310.xml; do
  cp "$SRC/$f" "$DST/$(echo "$f" | tr / _).v136"
done
echo "copied to $DST"
