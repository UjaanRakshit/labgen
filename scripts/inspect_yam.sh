#!/usr/bin/env bash
Y=/home/ujaan/isaac/i2rt/i2rt/robot_models/arm/yam/v1
echo "--- dir ---"; ls -la "$Y" | head -12
echo "--- mesh files referenced by the urdf ---"
grep -o 'filename="[^"]*"' "$Y/yam.urdf" | sort -u | head -15
echo "--- mesh files on disk ---"
find "$Y/.." -name '*.stl' -o -name '*.obj' -o -name '*.dae' 2>/dev/null | head -15
echo "--- joint count ---"
echo "urdf joints: $(grep -c '<joint' "$Y/yam.urdf")"
echo "mjcf joints: $(grep -c '<joint' "$Y/yam.xml")"
echo "--- joint names ---"
grep -o '<joint name="[^"]*"' "$Y/yam.urdf" | head -12
