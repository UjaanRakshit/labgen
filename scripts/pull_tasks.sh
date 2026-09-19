#!/usr/bin/env bash
SRC=/home/ujaan/isaac/labgen/frames_tasks
DST="/mnt/c/Ujaan Docx/Research/labgen/out/frames_tasks"
rm -rf "$DST"; mkdir -p "$DST"
cp "$SRC"/*.png "$DST"/
echo "copied $(ls "$DST" | wc -l) frames"
