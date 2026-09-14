#!/usr/bin/env bash
# What does this Isaac Lab checkout actually need in order to load a USD scene?
#
# The question that decides the install size: CLAUDE.md says "Isaac Sim 6.0 is
# optional if you only use the kit-less Newton path". If true, we avoid a ~15 GB
# Isaac Sim download. If Isaac Lab's scene loading still goes through omni.*,
# it is not optional and we need the whole thing.
set -uo pipefail
LAB="$HOME/isaac/IsaacLab"
cd "$LAB" || exit 1

say() { printf '\n=== %s ===\n' "$1"; }

say "pinned commit"
/usr/bin/git -C "$LAB" log -1 --format='%h %cs %s'
echo "version: $(cat "$LAB/VERSION")"

say "install extras declared in pyproject"
sed -n '/\[project.optional-dependencies\]/,/^\[/p' pyproject.toml | head -40

say "newton-related packages required"
grep -rn "newton\|mujoco" pyproject.toml source/isaaclab/setup.py 2>/dev/null | head -20

say "is there a kit-less / newton-only entry point?"
ls -d source/* 2>/dev/null
echo "--- newton backend module ---"
find source -ipath "*newton*" -name "*.py" 2>/dev/null | head -15

say "does InteractiveScene depend on omni/isaacsim?"
SCENE="source/isaaclab/isaaclab/scene/interactive_scene.py"
if [ -f "$SCENE" ]; then
    grep -n "^import\|^from" "$SCENE" | head -25
else
    echo "  not found at $SCENE"
    find source -name "interactive_scene.py" 2>/dev/null | head -3
fi

say "spawner: how does a USD file get onto a stage?"
find source -name "from_files.py" -path "*spawners*" 2>/dev/null | head -2
FF=$(find source -name "from_files.py" -path "*spawners*" 2>/dev/null | head -1)
[ -n "$FF" ] && grep -n "^import\|^from\|def spawn_from_usd\|def spawn_ground" "$FF" | head -20

say "renderer configs mentioned in CLAUDE.md"
grep -rn "NewtonWarpRendererCfg\|IsaacRtxRendererCfg" source --include=*.py 2>/dev/null | head -8
