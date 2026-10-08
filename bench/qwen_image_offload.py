"""Qwen-Image-2.1 through diffusers with model CPU offload on MPS, one image, with a memory trace.

The same prompt, size, steps, seed, weights, torch, diffusers and stageload as the eager and
staged runs of 2026-10-06 (bench/results/2026-10-06-qwen-image), so the offload peak can sit
next to theirs: eager 43.6 GB (stopped at decode, swap +8 GB), staged 19.0 GB.

    python bench/qwen_image_offload.py --out runs/offload-1.png --steps 20
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from stageload import Event, MemoryMeter, __version__, on_call

REPO = "Qwen/Qwen-Image-2.1"
DTYPE = torch.bfloat16
PROMPT = (
    "A small wooden workbench by a window, a hand plane and wood shavings on it, "
    "morning light, photograph"
)
STAGES = (
    ("encode_prompt", "encode"), ("prepare_latents", "denoise"), ("_unpack_latents", "decode"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    import diffusers
    from diffusers import QwenImage21Pipeline

    trace = args.out.with_suffix(".trace.jsonl")
    meta = {
        "mode": "offload", "pipeline": "Qwen-Image-2.1", "steps": args.steps, "size": args.size,
        "seed": args.seed, "stageload": __version__, "torch": torch.__version__,
        "diffusers": diffusers.__version__, "offload": "enable_model_cpu_offload(device='mps')",
    }
    with MemoryMeter(trace, meta=meta) as meter:
        start = time.monotonic()
        pipe = QwenImage21Pipeline.from_pretrained(REPO, torch_dtype=DTYPE)
        pipe.enable_model_cpu_offload(device="mps")
        loaded = time.monotonic() - start
        meter.emit(Event("mark", note=f"pipeline loaded with model CPU offload in {loaded:.1f} s"))
        for method, stage in STAGES:
            def mark(*a, _s=stage, **k):
                meter.emit(Event("stage", stage=_s))

            on_call(pipe, method, before=mark)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        with torch.no_grad():
            image = pipe(prompt=PROMPT, width=args.size, height=args.size,
                         num_inference_steps=args.steps, generator=generator).images[0]
    image.save(args.out)
    summary = meter.summary()
    peak = summary["peak_footprint"] / 2**30
    print(f"qwen-image: {args.out.name} | offload | peak footprint {peak:.1f} GB | "
          f"{summary['duration']:.0f} s | trace {trace.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
