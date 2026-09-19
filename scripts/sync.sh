#!/usr/bin/env bash
# Copy labgen's scripts/ into the WSL working dir, stripping CRLF.
# Inline `bash -c` from the Windows side mangles variable expansion, so every
# multi-step WSL action lives in a file like this one.
SRC="/mnt/c/Ujaan Docx/Research/labgen/scripts"
DST=/home/ujaan/isaac/labgen
mkdir -p "$DST"
n=0
for f in "$SRC"/*.py "$SRC"/*.sh; do
    [ -f "$f" ] || continue
    sed 's/\r$//' "$f" > "$DST/$(basename "$f")"
    n=$((n+1))
done
echo "synced $n files to $DST"
