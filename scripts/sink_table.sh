#!/usr/bin/env bash
# The same vessel table before and after the contact fix, so the thin-wall
# hypothesis can be read off directly rather than inferred from one scene.
set -u
PY=/home/ujaan/isaac/IsaacLab/.venv/bin/python
cd /home/ujaan/isaac/labgen || exit 1
{
  echo "######## BEFORE: import defaults ########"
  "$PY" probe_sink.py 2>/dev/null
  echo
  echo "######## AFTER: ke=160000, dt=1/960 ########"
  "$PY" probe_sink.py 160000 960 2>/dev/null
} | tee /tmp/sink_table.log
