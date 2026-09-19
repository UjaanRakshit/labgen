#!/usr/bin/env bash
R=/home/ujaan/isaac/i2rt/i2rt/robot_models
echo "=== models with collision geoms ==="
for f in $(find "$R" -name '*.xml'); do
  nc=$(grep -c 'class="collision"\|contype\|conaffinity' "$f" 2>/dev/null)
  ng=$(grep -c '<geom' "$f" 2>/dev/null)
  echo "$(basename $(dirname $f))/$(basename $f)  geoms=$ng collision_markers=$nc"
done
echo
echo "=== yam.xml geom classes ==="
grep -o 'class="[a-z]*"' "$R/arm/yam/v1/yam.xml" | sort | uniq -c
echo
echo "=== default classes in yam.xml ==="
sed -n '/<default/,/<\/default>/p' "$R/arm/yam/v1/yam.xml" | head -25
echo
echo "=== gripper dir ==="
ls -R "$R/gripper" 2>/dev/null | head -20
