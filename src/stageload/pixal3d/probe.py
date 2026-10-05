"""Fingerprints of what each sampler produced, and stopping a run where a stage begins.

Fingerprints answer "did staged loading change the result?" without a finished run: every
flow stage's output is hashed and saved, so two runs can be compared stage by stage. Stopping
at a stage lets a run that would not fit in memory still deliver the stages before it.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch

from ..hooks import on_call

SAMPLERS = ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler")


class StopRun(BaseException):
    """Raised where a stage begins when the run was asked to stop there.

    A BaseException, like KeyboardInterrupt, so that an ``except Exception`` in the port does
    not catch it and carry on.
    """

    def __init__(self, stage: str) -> None:
        super().__init__(f"stopped when entering stage {stage!r}")
        self.stage = stage


def stopping(on_stage: Callable[[str], None], stop_at: str | None) -> Callable[[str], None]:
    """``on_stage`` that raises :class:`StopRun` instead of entering ``stop_at``."""
    if stop_at is None:
        return on_stage

    def guarded(stage: str) -> None:
        if stage == stop_at:
            raise StopRun(stage)
        on_stage(stage)

    return guarded


def _tensors(obj: Any) -> dict[str, torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return {"tensor": obj}
    if hasattr(obj, "coords") and hasattr(obj, "feats"):
        return {"coords": obj.coords, "feats": obj.feats}
    if isinstance(obj, Mapping) and "samples" in obj:
        return _tensors(obj["samples"])
    samples = getattr(obj, "samples", None)
    return _tensors(samples) if samples is not None else {}


def _describe(name: str, tensor: torch.Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to("cpu")
    if cpu.dtype == torch.bfloat16:
        cpu = cpu.float()
    data = cpu.contiguous().numpy().tobytes()
    return {
        "name": name,
        "shape": list(cpu.shape),
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


class Fingerprints:
    """Hash and save the output of every ``sample`` call of the pipeline's samplers."""

    def __init__(self, out_dir: str | Path) -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []
        self._counts: dict[str, int] = {}

    def install(self, pipeline: Any) -> None:
        for attr in SAMPLERS:
            sampler = getattr(pipeline, attr, None)
            if sampler is not None and hasattr(sampler, "sample"):
                on_call(sampler, "sample", after=self._recorder(attr))

    def _recorder(self, attr: str) -> Callable[..., None]:
        def after(result: Any, *args: Any, **kwargs: Any) -> None:
            self._counts[attr] = self._counts.get(attr, 0) + 1
            call = f"{attr}#{self._counts[attr]}"
            tensors = _tensors(result)
            self.records.append(
                {"call": call, "tensors": [_describe(k, v) for k, v in tensors.items()]}
            )
            torch.save({k: v.detach().to("cpu") for k, v in tensors.items()},
                       self.out_dir / f"{call}.pt")
            (self.out_dir / "fingerprints.json").write_text(
                json.dumps(self.records, indent=2) + "\n"
            )

        return after
