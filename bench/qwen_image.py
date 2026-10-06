"""Qwen-Image-2.1 through diffusers, one image, eager or staged, with a memory trace.

Eager is the usual ``QwenImage21Pipeline.from_pretrained(...).to("mps")``: the text encoder
(17.5 GB), the transformer (14.2 GB) and the VAE (1.4 GB) stay in memory for the whole run.
Staged keeps the VAE and loads the two large models one at a time: the text encoder for
``encode_prompt``, the transformer from ``prepare_latents`` on, and neither during the VAE
decode. The pipeline's code is unchanged; its two large models are attributes that answer
``.config`` from the checkpoint and load the model on any other use.

    python bench/qwen_image.py --mode staged --out runs/staged-1.png --steps 20

needs diffusers from main (QwenImage21Pipeline) and the weights in the Hugging Face cache.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch

from stageload import Event, MemoryMeter, StagedModels, __version__, on_call

REPO = "Qwen/Qwen-Image-2.1"
DEVICE = torch.device("mps")
DTYPE = torch.bfloat16
STAGES = {"encode": ["text_encoder"], "denoise": ["transformer"], "decode": []}
PROMPT = (
    "A small wooden workbench by a window, a hand plane and wood shavings on it, "
    "morning light, photograph"
)


class LazyModel:
    """Stands in a pipeline attribute for a model that StagedModels loads on first real use.

    ``config`` comes from the checkpoint without loading weights, because the pipeline reads
    the transformer's config before the denoising stage begins. Private names raise
    AttributeError instead of loading, so probes such as ``hasattr(model, "_hf_hook")`` stay
    cheap; everything else, including calls, goes to the loaded model.
    """

    def __init__(self, models: StagedModels, name: str, config: Any) -> None:
        object.__setattr__(self, "_models", models)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_config", config)

    def __getattr__(self, attr: str) -> Any:
        if attr == "config":
            return self._config
        if attr.startswith("_"):
            raise AttributeError(attr)
        return getattr(self._models[self._name], attr)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._models[self._name](*args, **kwargs)


def install_stage_hooks(pipe: Any, on_stage: Any) -> None:
    """Call ``on_stage(name)`` where each stage of ``QwenImage21Pipeline.__call__`` begins."""
    on_call(pipe, "encode_prompt", before=lambda *a, **k: on_stage("encode"))
    on_call(pipe, "prepare_latents", before=lambda *a, **k: on_stage("denoise"))
    on_call(pipe, "_unpack_latents", before=lambda *a, **k: on_stage("decode"))


def load_eager(sink: MemoryMeter) -> Any:
    from diffusers import QwenImage21Pipeline

    start = time.monotonic()
    pipe = QwenImage21Pipeline.from_pretrained(REPO, torch_dtype=DTYPE).to(DEVICE)
    sink.emit(Event("mark", note=f"pipeline loaded in {time.monotonic() - start:.1f} s"))
    return pipe


def load_staged(sink: MemoryMeter) -> tuple[Any, StagedModels]:
    from diffusers import (
        AutoencoderKLQwenImage21,
        QwenImage21Pipeline,
        QwenImage21Transformer2DModel,
    )
    from diffusers.configuration_utils import FrozenDict
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration

    def text_encoder() -> torch.nn.Module:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            REPO, subfolder="text_encoder", torch_dtype=DTYPE
        )
        return model.to(DEVICE).eval()

    def transformer() -> torch.nn.Module:
        model = QwenImage21Transformer2DModel.from_pretrained(
            REPO, subfolder="transformer", torch_dtype=DTYPE
        )
        return model.to(DEVICE).eval()

    models = StagedModels(
        {"text_encoder": text_encoder, "transformer": transformer}, stages=STAGES, events=sink
    )
    vae = AutoencoderKLQwenImage21.from_pretrained(REPO, subfolder="vae", torch_dtype=DTYPE)
    pipe = QwenImage21Pipeline.from_pretrained(
        REPO, text_encoder=None, transformer=None, vae=vae.to(DEVICE), torch_dtype=DTYPE
    )
    configs = {
        "text_encoder": AutoConfig.from_pretrained(REPO, subfolder="text_encoder"),
        "transformer": FrozenDict(
            QwenImage21Transformer2DModel.load_config(REPO, subfolder="transformer")
        ),
    }
    for name, config in configs.items():
        # past DiffusionPipeline.__setattr__, which would register the stand-in as a component
        object.__setattr__(pipe, name, LazyModel(models, name, config))
    return pipe, models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--mode", choices=("eager", "staged"), required=True)
    parser.add_argument(
        "--out", type=Path, required=True, help="output PNG; the trace goes next to it"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--prompt", default=PROMPT)
    args = parser.parse_args(argv)

    import diffusers

    args.out.parent.mkdir(parents=True, exist_ok=True)
    trace = args.out.with_suffix(".trace.jsonl")
    meta = {
        "mode": args.mode,
        "pipeline": "Qwen-Image-2.1",
        "steps": args.steps,
        "size": args.size,
        "seed": args.seed,
        "stageload": __version__,
        "torch": torch.__version__,
        "diffusers": diffusers.__version__,
    }
    with MemoryMeter(trace, meta=meta) as meter:
        models: StagedModels | None = None
        if args.mode == "staged":
            pipe, models = load_staged(meter)
            on_stage = models.enter
        else:
            pipe = load_eager(meter)

            def on_stage(stage: str) -> None:
                meter.emit(Event("stage", stage=stage))

        install_stage_hooks(pipe, on_stage)
        generator = torch.Generator("cpu").manual_seed(args.seed)
        with torch.no_grad():
            image = pipe(
                prompt=args.prompt,
                width=args.size,
                height=args.size,
                num_inference_steps=args.steps,
                generator=generator,
            ).images[0]
        if models is not None:
            models.close()
    image.save(args.out)
    summary = meter.summary()
    print(
        f"qwen-image: {args.out.name} | {args.mode} | peak footprint "
        f"{summary['peak_footprint'] / 2**30:.1f} GB | {summary['duration']:.0f} s | "
        f"trace {trace.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
