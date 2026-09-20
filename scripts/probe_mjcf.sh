#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "--- AssetConverterBaseCfg fields ---"
grep -nE '^\s{4}[a-z_]+:' "$L/sim/converters/asset_converter_base_cfg.py" | head -8
echo "--- MjcfConverterCfg fields ---"
grep -nE '^\s{4}[a-z_]+:' "$L/sim/converters/mjcf_converter_cfg.py" | head -10
echo "--- MjcfFileCfg body ---"
sed -n '303,330p' "$L/sim/spawners/from_files/from_files_cfg.py" | grep -vE '^\s*#|^\s*$' | head -12
