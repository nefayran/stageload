"""Fill MPS memory in 512 MiB steps until PyTorch refuses, and print what it said.

Run once per PYTORCH_MPS_HIGH_WATERMARK_RATIO setting, each in a fresh process, under `stageload run` with a swap
budget: with the ratio at 0.0 PyTorch sets no limit, and the guard is what stops the run.
Output: one JSON line per step and a final line with the error text, to stdout.
"""
import json
import os
import time

import torch

GIB = 1024 ** 3
STEP = 512 * 1024 ** 2          # bytes per tensor
CAP = 46 * GIB                  # never ask for more than this in total, whatever the ratio

t0 = time.time()
rec = torch.mps.recommended_max_memory()
print(json.dumps({"ratio": os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "default"),
                  "low": os.environ.get("PYTORCH_MPS_LOW_WATERMARK_RATIO", "default"),
                  "recommended_max_gib": round(rec / GIB, 2), "torch": torch.__version__}), flush=True)
held = []
try:
    while len(held) * STEP < CAP:
        # random values, not ones: macOS compresses pages of identical values almost to nothing, and the first run
        # (13.10, default ratio) reached 46 GiB of ones with no error and no swap, which says nothing about model weights
        held.append(torch.rand(STEP // 2, dtype=torch.float16, device="mps"))
        torch.mps.synchronize()
        print(json.dumps({"t": round(time.time() - t0, 2), "tensors_gib": round(len(held) * STEP / GIB, 2),
                          "allocated_gib": round(torch.mps.current_allocated_memory() / GIB, 2),
                          "driver_gib": round(torch.mps.driver_allocated_memory() / GIB, 2)}), flush=True)
        time.sleep(0.5)          # let the guard sample between steps
    print(json.dumps({"end": "cap reached", "tensors_gib": round(len(held) * STEP / GIB, 2)}), flush=True)
except RuntimeError as e:
    print(json.dumps({"end": "RuntimeError", "t": round(time.time() - t0, 2),
                      "tensors_gib": round(len(held) * STEP / GIB, 2), "error": str(e)}), flush=True)
