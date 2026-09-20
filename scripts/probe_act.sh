#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== actuator cfg classes ==="
grep -rn "^class \|^@configclass" "$L/actuators/actuator_cfg.py" | head -14
echo
echo "=== ImplicitActuatorCfg fields ==="
grep -n -A 12 "class ImplicitActuatorCfg" "$L/actuators/actuator_cfg.py" | head -20
echo
echo "=== ActuatorBaseCfg fields ==="
grep -nE "^\s{4}(joint_names_expr|effort_limit|velocity_limit|stiffness|damping|armature|friction)" "$L/actuators/actuator_cfg.py" | head -14
echo
echo "=== UsdFileCfg key fields ==="
grep -n -A 20 "^class UsdFileCfg" "$L/sim/spawners/from_files/from_files_cfg.py" | grep -E "usd_path|variants|class |\"\"\"" | head -8
