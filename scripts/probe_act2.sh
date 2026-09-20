#!/usr/bin/env bash
L=/home/ujaan/isaac/IsaacLab/source/isaaclab/isaaclab
echo "=== actuators package ==="; ls "$L/actuators"
echo; echo "=== exported names ==="
grep -E "^from|^\s+\"" "$L/actuators/__init__.py" | head -20
echo; echo "=== ImplicitActuatorCfg ==="
F=$(grep -rl "class ImplicitActuatorCfg" "$L/actuators")
echo "in: $F"
grep -n -A 10 "class ImplicitActuatorCfg" "$F" | head -16
echo; echo "=== ActuatorBaseCfg core fields ==="
G=$(grep -rl "class ActuatorBaseCfg" "$L/actuators")
grep -nE "^\s{4}[a-z_]+:" "$G" | head -16
