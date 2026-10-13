#!/bin/zsh
# Four fills, one stageload queue ticket each, so a guard stop on one run does not lose the others.
# Results: bench/results/2026-10-13-mps-watermark/<ratio>.jsonl, the guard's trace next to it.
set -u
D=${0:A:h}; R=${D:h}/results/2026-10-13-mps-watermark; PY=$HOME/local-llm/qwenimage-venv/bin/python
for ratio in 0.5 1.0 default 0.0; do
  if [[ $ratio == default ]]; then envs=(); else envs=(PYTORCH_MPS_HIGH_WATERMARK_RATIO=$ratio); fi
  $PY -m stageload run --who claude-mps-watermark-$ratio --lane ram --swap-budget 2G --trace $R/$ratio.trace.jsonl \
    -- nice -n 15 env $envs $PY $D/fill.py > $R/$ratio.jsonl 2> $R/$ratio.stderr.txt
  echo "$ratio exit $?" >> $R/runs.txt
done
echo done >> $R/runs.txt
