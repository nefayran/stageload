"""A model registry that loads models on first use and frees them between stages."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

from torch import nn

from .events import Event, EventSink
from .release import ReleasedModuleError, module_nbytes, release
from .rng import preserved_rng

Loader = Callable[[], nn.Module]


@dataclass
class _Slot:
    loader: Loader | None
    module: nn.Module | None = None
    released_at: str | None = None
    loads: int = 0


class StagedModels(MutableMapping[str, nn.Module]):
    """Holds a pipeline's models; only the models the current stage needs stay in memory.

    ``models[name]`` loads a model the first time it is asked for. ``enter(stage)`` releases
    every loaded model that the stage does not list and that is not pinned. Membership tests,
    ``len`` and iteration over names never load anything, and ``values()``/``items()`` only
    yield models that are loaded right now.

    With ``preserve_rng`` (the default) a load leaves the CPU, MPS and CUDA random generators
    where they were, so a model built in the middle of a seeded run does not change the noise the
    run draws next.
    """

    def __init__(
        self,
        loaders: Mapping[str, Loader],
        *,
        stages: Mapping[str, Iterable[str]],
        pinned: Iterable[str] = (),
        events: EventSink | None = None,
        preserve_rng: bool = True,
    ) -> None:
        self._preserve_rng = preserve_rng
        self._slots = {name: _Slot(loader) for name, loader in loaders.items()}
        self._stages = {stage: frozenset(names) for stage, names in stages.items()}
        unknown = sorted({n for names in self._stages.values() for n in names} - self._slots.keys())
        if unknown:
            raise KeyError(f"stages name models that have no loader: {unknown}")
        self._pinned = set(pinned)
        unknown_pins = sorted(self._pinned - self._slots.keys())
        if unknown_pins:
            raise KeyError(f"pinned models have no loader: {unknown_pins}")
        self._events = events
        self.current_stage: str | None = None

    def __getitem__(self, name: str) -> nn.Module:
        slot = self._slots[name]
        if slot.module is not None:
            return slot.module
        if slot.loader is None:
            raise ReleasedModuleError(name, slot.released_at)
        kind = "reload" if slot.loads else "load"
        start = time.monotonic()
        if self._preserve_rng:
            with preserved_rng():
                module = slot.loader()
        else:
            module = slot.loader()
        seconds = time.monotonic() - start
        slot.module = module
        slot.loads += 1
        outside = self.current_stage is not None and name not in self._stages[self.current_stage]
        self._emit(
            kind,
            name=name,
            nbytes=module_nbytes(module),
            seconds=round(seconds, 3),
            note="outside_stage" if outside else None,
        )
        return module

    def __setitem__(self, name: str, module: nn.Module) -> None:
        if name in self._slots and self._slots[name].module is not None:
            self._release(name)
        self._slots[name] = _Slot(loader=None, module=module, loads=1)

    def __delitem__(self, name: str) -> None:
        if self._slots[name].module is not None:
            self._release(name)
        del self._slots[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._slots)

    def __len__(self) -> int:
        return len(self._slots)

    def __contains__(self, name: object) -> bool:
        return name in self._slots

    def values(self) -> list[nn.Module]:  # type: ignore[override]
        return [slot.module for slot in self._slots.values() if slot.module is not None]

    def items(self) -> list[tuple[str, nn.Module]]:  # type: ignore[override]
        return [(n, s.module) for n, s in self._slots.items() if s.module is not None]

    def loaded(self) -> list[str]:
        return [name for name, slot in self._slots.items() if slot.module is not None]

    def enter(self, stage: str) -> None:
        if stage not in self._stages:
            raise KeyError(f"unknown stage {stage!r}; known stages: {sorted(self._stages)}")
        self.current_stage = stage
        self._emit("stage")
        needed = self._stages[stage]
        for name in self.loaded():
            if name not in needed and name not in self._pinned:
                self._release(name)

    def pin(self, name: str) -> None:
        if name not in self._slots:
            raise KeyError(name)
        self._pinned.add(name)

    def unpin(self, name: str) -> None:
        self._pinned.discard(name)

    def close(self) -> None:
        for name in self.loaded():
            self._release(name)
        self.current_stage = None

    def _release(self, name: str) -> None:
        slot = self._slots[name]
        module = slot.module
        if module is None:
            return
        others = [s.module for n, s in self._slots.items() if n != name and s.module is not None]
        stats = release(module, name=name, stage=self.current_stage, keep=others)
        slot.module = None
        slot.released_at = self.current_stage
        self._emit(
            "release",
            name=name,
            nbytes=stats.nbytes,
            note=f"kept={stats.kept}" if stats.kept else None,
        )

    def _emit(self, kind: str, **fields: Any) -> None:
        if self._events is not None:
            self._events.emit(Event(kind, stage=self.current_stage, **fields))
