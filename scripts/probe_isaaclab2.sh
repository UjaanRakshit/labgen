#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== RigidObjectCfg fields ==="
grep -vE '^\s*#|^$' "$L/assets/rigid_object/rigid_object_cfg.py" | sed -n '1,40p'
echo
echo "=== AssetBaseCfg fields ==="
grep -vE '^\s*#|^$' "$L/assets/asset_base_cfg.py" | sed -n '10,55p'
echo
echo "=== UsdFileCfg / spawners ==="
grep -n "class UsdFileCfg" -A 14 "$L/sim/spawners/from_files/from_files_cfg.py" | head -22
