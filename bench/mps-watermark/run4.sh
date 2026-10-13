#!/bin/zsh
# One more point between the clean error at 0.5 and the swap at 1.0.
set -u
D=${0:A:h}; R=${D:h}/results/2026-10-13-mps-watermark; PY=$HOME/local-llm/qwenimage-venv/bin/python
$PY -m stageload run --who claude-mps-watermark-0.8-both --lane ram --swap-budget 2G --trace $R/0.8.trace.jsonl \
  -- nice -n 15 env PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.8 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.8 $PY $D/fill.py > $R/0.8.jsonl 2> $R/0.8.stderr.txt
echo "0.8 (high and low) exit $?" >> $R/runs.txt
