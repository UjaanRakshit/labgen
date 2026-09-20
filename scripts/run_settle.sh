#!/usr/bin/env bash
cd /home/ujaan/isaac/labgen
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
for s in bench_5 bench_arm; do
  $PY -c "
import sys; sys.path.insert(0,'/mnt/c/Ujaan Docx/Research/labgen')
from labgen.settle import settle
print(settle('/home/ujaan/isaac/labgen/$s.usda'))
print()
" 2>/dev/null | grep -vE '^Module|^Warp|CUDA|Devices|cuda:0|cpu|Kernel|cache'
done
