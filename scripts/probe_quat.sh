#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== AssetBaseCfg.InitialStateCfg rot declaration ==="
grep -n -B2 -A6 'rot: tuple\[float, float, float, float\]' "$L/assets/asset_base_cfg.py" | head -20
echo
echo "=== how rot is consumed ==="
grep -rn "init_state.rot" "$L/assets" | head -8
echo
echo "=== ArticulationCfg ==="
grep -vE '^\s*#|^$' "$L/assets/articulation/articulation_cfg.py" | sed -n '1,45p'
