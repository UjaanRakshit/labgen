#!/usr/bin/env bash
# Stage B: install Isaac Lab 3.0 Beta (kit-less Newton path) into its own venv.
#
# Kit-less on purpose. The base dependency list in the pinned commit contains no
# isaacsim-* packages -- those live only in the `importers` extra -- so the
# default resolve gives Newton + warp + torch + usd-exchange and skips the ~15 GB
# Isaac Sim download. CLAUDE.md said Isaac Sim 6.0 is optional on this path and
# that is borne out by the metadata.
#
# NOTE: usd-exchange is this environment's ONLY pxr provider. Isaac Lab's own
# comment says two providers overwrite each other's files and removing either
# then breaks pxr. Do not add usd-core here; labgen keeps it in a separate venv.
set -uo pipefail
export PATH="$HOME/.local/bin:$PATH"
LAB="$HOME/isaac/IsaacLab"
LOG="$HOME/isaac/stage_b.log"

exec > >(tee "$LOG") 2>&1

echo "=== purge stale wheels (CLAUDE.md: known conflicts) ==="
# mujoco / mujoco-warp / newton wheels cached from an earlier resolve are the
# documented failure mode here.
for pkg in mujoco mujoco-warp mujoco_warp newton newton-sim; do
    uv cache clean "$pkg" 2>&1 | tail -1
done

echo
echo "=== pinned commit ==="
/usr/bin/git -C "$LAB" log -1 --format='%h %cs %s'

echo
echo "=== uv sync (kit-less newton) ==="
cd "$LAB" || exit 1
start=$(date +%s)
uv sync --no-progress
rc=$?
end=$(date +%s)
echo
echo "uv sync exit=$rc  elapsed=$(( (end-start)/60 ))m$(( (end-start)%60 ))s"
[ $rc -ne 0 ] && { echo "STAGE B FAILED"; exit $rc; }

echo
echo "=== installed size ==="
du -sh "$LAB/.venv" 2>/dev/null

echo
echo "=== import smoke test ==="
"$LAB/.venv/bin/python" - <<'PY'
import importlib, sys
print("python:", sys.version.split()[0])
for mod in ("torch", "warp", "newton", "pxr", "isaaclab"):
    try:
        m = importlib.import_module(mod)
        print(f"  {mod:10} OK  {getattr(m, '__version__', '')}")
    except Exception as e:
        print(f"  {mod:10} FAIL  {type(e).__name__}: {str(e)[:120]}")
try:
    import torch
    print("  cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("  device:", torch.cuda.get_device_name(0))
        print("  capability:", torch.cuda.get_device_capability(0))
except Exception as e:
    print("  cuda check failed:", e)
PY
echo
echo "=== STAGE B COMPLETE ==="
