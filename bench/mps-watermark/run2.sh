#!/bin/zsh
# Repeat of the 0.5 and 1.0 fills with the low watermark set too: with only the high ratio below the default low
# ratio (1.4), MPS refuses to start ("invalid low watermark ratio 1.4"), see <ratio>-high-only.stderr.txt.
set -u
D=${0:A:h}; R=${D:h}/results/2026-10-13-mps-watermark; PY=$HOME/local-llm/qwenimage-venv/bin/python
for ratio in 0.5 1.0; do
  $PY -m stageload run --who claude-mps-watermark-$ratio-both --lane ram --swap-budget 2G --trace $R/$ratio.trace.jsonl \
    -- nice -n 15 env PYTORCH_MPS_HIGH_WATERMARK_RATIO=$ratio PYTORCH_MPS_LOW_WATERMARK_RATIO=$ratio $PY $D/fill.py > $R/$ratio.jsonl 2> $R/$ratio.stderr.txt
  echo "$ratio (high and low) exit $?" >> $R/runs.txt
done
echo done2 >> $R/runs.txt
