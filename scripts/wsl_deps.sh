#!/usr/bin/env bash
set -uo pipefail
LAB="$HOME/isaac/IsaacLab"
say() { printf '\n=== %s ===\n' "$1"; }

say "base dependencies (lines 20-120)"
sed -n '20,120p' "$LAB/pyproject.toml"

say "dependency-groups / physx vs newton split"
grep -n "isaaclab-physx\|isaaclab-newton\|isaaclab-ov\|isaacsim" "$LAB/pyproject.toml" | head -30

say "uv sources"
sed -n '/\[tool.uv.sources\]/,/^\[/p' "$LAB/pyproject.toml" | head -30

say "has_kit"
cat "$LAB/source/isaaclab/isaaclab/utils/version.py" 2>/dev/null | head -40
