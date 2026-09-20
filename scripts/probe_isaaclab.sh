#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== InteractiveSceneCfg ==="
sed -n '1,60p' "$L/scene/interactive_scene_cfg.py" | grep -vE '^#|^$' | head -30
echo
echo "=== asset cfg classes available ==="
ls "$L/assets"
echo
echo "=== a worked example scene cfg ==="
find /home/ujaan/isaac/IsaacLab/scripts -name '*.py' 2>/dev/null | xargs grep -ln "InteractiveSceneCfg" 2>/dev/null | head -3
