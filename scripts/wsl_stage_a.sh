#!/usr/bin/env bash
# Stage A of the Isaac Lab bring-up: cheap, fast, fails early.
#
# Installs uv, clones Isaac Lab at a PINNED commit on the develop branch (never
# the tip -- CLAUDE.md), and reports what the environment looks like. Does NOT
# download Isaac Sim; that is stage B and it is ~15 GB.
set -uo pipefail

ROOT="$HOME/isaac"
LAB="$ROOT/IsaacLab"
mkdir -p "$ROOT"

say() { printf '\n=== %s ===\n' "$1"; }

say "uv"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version || { echo "FATAL: uv did not install"; exit 1; }

say "python 3.12"
# Ubuntu 26.04 ships 3.14 and has no python3.12 in apt. uv supplies it, which
# keeps us on the version Isaac Lab 3.0 Beta 2 expects without touching the
# system interpreter.
uv python install 3.12 2>&1 | tail -2
uv python find 3.12 || { echo "FATAL: no python 3.12"; exit 1; }

say "clone Isaac Lab (develop)"
if [ ! -d "$LAB/.git" ]; then
    git clone --branch develop https://github.com/isaac-sim/IsaacLab.git "$LAB" 2>&1 | tail -3
fi
cd "$LAB" || exit 1
git fetch origin develop --quiet 2>&1 | tail -2
echo "branch tip: $(git rev-parse --short origin/develop)  $(git log -1 --format=%cs origin/develop)"

say "environment vs CLAUDE.md"
printf '  %-22s %s\n' "required OS:" "Ubuntu 22.04 or 24.04"
printf '  %-22s %s\n' "actual OS:" "$(lsb_release -ds 2>/dev/null)"
printf '  %-22s %s\n' "glibc:" "$(ldd --version | head -1 | awk '{print $NF}')"
printf '  %-22s %s\n' "GPU:" "$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo none)"
printf '  %-22s %s\n' "VRAM:" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo n/a)"
printf '  %-22s %s\n' "RAM:" "$(free -g | awk 'NR==2{print $2" GB"}')"
printf '  %-22s %s\n' "disk free:" "$(df -h "$HOME" | awk 'NR==2{print $4}')"

say "stale wheels in the uv/pip cache"
# CLAUDE.md: known dependency conflicts with stale mujoco / mujoco-warp /
# newton wheels. Report before installing anything.
found=0
for pat in mujoco mujoco-warp mujoco_warp newton; do
    hits=$(find "$HOME/.cache/uv" "$HOME/.cache/pip" -iname "*${pat}*" 2>/dev/null | head -3)
    if [ -n "$hits" ]; then echo "  $pat:"; echo "$hits" | sed 's/^/    /'; found=1; fi
done
[ "$found" -eq 0 ] && echo "  none (clean cache)"

say "stage A complete"
echo "Isaac Lab checked out at: $LAB"
echo "Nothing large downloaded yet."
