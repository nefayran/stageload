"""Events that stageload emits while it loads, releases and changes stages."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

EVENT_KINDS = ("load", "reload", "release", "stage", "mark")


@dataclass(frozen=True)
class Event:
    kind: str
    name: str | None = None
    stage: str | None = None
    nbytes: int | None = None
    seconds: float | None = None
    note: str | None = None
    t: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise ValueError(f"unknown event kind {self.kind!r}; expected one of {EVENT_KINDS}")

    def as_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class EventSink(Protocol):
    def emit(self, event: Event) -> None: ...


class ListSink:
    """Keeps events in a list, for tests and quick summaries."""

    def __init__(self) -> None:
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
