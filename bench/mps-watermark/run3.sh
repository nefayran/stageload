#!/bin/zsh
# The default-ratio fill again, with random values (see the note in fill.py); the first run with ones is default-ones.*.
set -u
D=${0:A:h}; R=${D:h}/results/2026-10-13-mps-watermark; PY=$HOME/local-llm/qwenimage-venv/bin/python
$PY -m stageload run --who claude-mps-watermark-default-rand --lane ram --swap-budget 2G --trace $R/default.trace.jsonl \
  -- nice -n 15 $PY $D/fill.py > $R/default.jsonl 2> $R/default.stderr.txt
echo "default (random) exit $?" >> $R/runs.txt
