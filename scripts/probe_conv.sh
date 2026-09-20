#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== converters ==="; ls "$L/sim/converters"
echo; echo "=== spawner file cfgs ==="
grep -n "^class .*FileCfg" "$L/sim/spawners/from_files/from_files_cfg.py"
echo; echo "=== does any spawner take urdf/mjcf directly? ==="
grep -rn "urdf\|mjcf" "$L/sim/spawners/from_files/from_files_cfg.py" | head -6
echo; echo "=== MjcfConverterCfg? ==="
grep -rn "class MjcfConverterCfg\|class UrdfConverterCfg" "$L/sim/converters"/*.py | head
