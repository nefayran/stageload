"""Free a module's weights in place, so references held elsewhere keep no memory alive.

On Apple Silicon the CPU and the GPU share one pool of memory, so moving a model to the CPU
frees nothing. Replacing its tensors with tensors on the ``meta`` device does.
"""

from __future__ import annotations

import gc
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import nn

_RELEASED = "_stageload_released"


class ReleasedModuleError(RuntimeError):
    """A module was called after stageload released its weights."""

    def __init__(self, name: str | None, stage: str | None) -> None:
        self.name = name
        self.stage = stage
        label = repr(name) if name else "This module"
        where = f" when entering stage {stage!r}" if stage else ""
        super().__init__(
            f"{label} was released{where} and its weights are gone. Get the model from the "
            "StagedModels registry again instead of keeping a reference to it."
        )


@dataclass(frozen=True)
class ReleaseStats:
    nbytes: int
    kept: int


def _storage_key(tensor: torch.Tensor) -> tuple[str, int]:
    return (tensor.device.type, tensor.untyped_storage().data_ptr())


def module_nbytes(module: nn.Module) -> int:
    """Bytes of storage held by a module's parameters and buffers, counting shared storage once."""
    seen: set[tuple[str, int]] = set()
    total = 0
    for tensor in [*module.parameters(), *module.buffers()]:
        if tensor.device.type == "meta":
            continue
        key = _storage_key(tensor)
        if key not in seen:
            seen.add(key)
            total += tensor.untyped_storage().nbytes()
    return total


def is_released(module: nn.Module) -> bool:
    return getattr(module, _RELEASED, None) is not None


def empty_device_cache() -> None:
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _raise_released(name: str | None, stage: str | None) -> Callable[..., None]:
    def forward(*args: object, **kwargs: object) -> None:
        raise ReleasedModuleError(name, stage)

    return forward


def release(
    module: nn.Module,
    *,
    name: str | None = None,
    stage: str | None = None,
    keep: Iterable[nn.Module] = (),
    empty_cache: bool = True,
) -> ReleaseStats:
    """Move every parameter and buffer of ``module`` to the meta device and free their storage.

    Tensors whose storage is shared with a module in ``keep`` are left alone. Afterwards the
    module's ``forward`` raises :class:`ReleasedModuleError`.
    """
    kept_storage = {
        _storage_key(tensor)
        for other in keep
        for tensor in [*other.parameters(), *other.buffers()]
        if tensor.device.type != "meta"
    }
    freed: set[tuple[str, int]] = set()
    nbytes = 0
    kept = 0
    for sub in module.modules():
        for table in (sub._parameters, sub._buffers):
            for key, tensor in list(table.items()):
                if tensor is None or tensor.device.type == "meta":
                    continue
                storage = _storage_key(tensor)
                if storage in kept_storage:
                    kept += 1
                    continue
                if storage not in freed:
                    freed.add(storage)
                    nbytes += tensor.untyped_storage().nbytes()
                meta = tensor.detach().to("meta")
                if table is sub._parameters:
                    table[key] = nn.Parameter(meta, requires_grad=tensor.requires_grad)
                else:
                    table[key] = meta
    setattr(module, _RELEASED, (name, stage))
    module.forward = _raise_released(name, stage)
    gc.collect()
    if empty_cache:
        empty_device_cache()
    return ReleaseStats(nbytes=nbytes, kept=kept)
