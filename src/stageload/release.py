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

    Tensors whose storage is shared with a module in ``keep`` are left alone. Afterwards calling
    ``module`` raises :class:`ReleasedModuleError`. Its submodules are released too, but calling
    one of them directly fails with PyTorch's own error about tensors on the meta device.
    """
    kept_storage = {
        _storage_key(tensor)
        for other in keep
        for tensor in [*other.parameters(), *other.buffers()]
        if tensor.device.type != "meta" and tensor.untyped_storage().nbytes()
    }
    freed: set[tuple[str, int]] = set()
    nbytes = 0
    kept = 0

    def frees(tensor: torch.Tensor | None) -> bool:
        nonlocal nbytes, kept
        if tensor is None or tensor.device.type == "meta":
            return False
        size = tensor.untyped_storage().nbytes()
        if not size:
            # empty storages all have the same address and hold nothing to keep or count
            return True
        storage = _storage_key(tensor)
        if storage in kept_storage:
            kept += 1
            return False
        if storage not in freed:
            freed.add(storage)
            nbytes += size
        return True

    for sub in module.modules():
        for key, param in list(sub._parameters.items()):
            if param is not None and frees(param):
                meta = param.detach().to("meta")
                sub._parameters[key] = nn.Parameter(meta, requires_grad=param.requires_grad)
        for key, buffer in list(sub._buffers.items()):
            if buffer is not None and frees(buffer):
                sub._buffers[key] = buffer.detach().to("meta")
    setattr(module, _RELEASED, (name, stage))
    module.forward = _raise_released(name, stage)
    gc.collect()
    if empty_cache:
        empty_device_cache()
    return ReleaseStats(nbytes=nbytes, kept=kept)
