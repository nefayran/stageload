"""What decides a waiting run's tier in its lane, besides its deadline and its owner's share.

Without these tiers a priority deadline on one owner's long batch put it before the runs of
everyone else, and a release whose slot was hours away waited a day. Before deadlines come:

1. a run for an item whose release slot is under ``STAGELOAD_NEAR_SLOT_H`` hours (36) away;
2. a run expected to take under ``STAGELOAD_SHORT_S`` seconds (900) that is not a MiniMax H3 video clip;
3. the rest in the usual order (deadline, then shares), where a group named in
   ``STAGELOAD_FLOORS`` (``thorn=20``: percent of the lane) takes the next turn while its part of
   the expected run time of the runs placed so far is below that.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from datetime import datetime
from pathlib import Path
from typing import Any

NEAR_SLOT_H = 36.0
SHORT_S = 900.0
FLOORS = "thorn=20"
# a real date: deadlines before 2002 are places (priority lines, the planner's tiers), not slots
_REAL_DATE = datetime(2002, 1, 1).timestamp()
_H3 = re.compile(r"(?<![a-z0-9])h3(?![0-9])|h3_shot|h3_exec|h3-modal|win-h3", re.I)
_KIND_CACHE: dict[Path, tuple[tuple[int, int], dict[tuple[str, str], float]]] = {}


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def near_slot_s() -> float:
    return _env_float("STAGELOAD_NEAR_SLOT_H", NEAR_SLOT_H) * 3600


def short_s() -> float:
    return _env_float("STAGELOAD_SHORT_S", SHORT_S)


def floors() -> list[tuple[str, float]]:
    """(share group, part of the lane 0..1) pairs from STAGELOAD_FLOORS, «thorn=20» by default."""
    out = []
    for part in os.environ.get("STAGELOAD_FLOORS", FLOORS).split(","):
        name, _, value = part.strip().partition("=")
        try:
            out.append((name.strip(), float(value) / 100))
        except ValueError:
            continue
    return [(g, f) for g, f in out if g and 0 < f < 1]


def _ts(value: Any) -> float | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def slot_of(data: dict[str, Any]) -> float | None:
    """The release slot the run serves: the planner's ``slot``, else a real --deadline."""
    for key in ("slot", "real_deadline", "deadline"):
        t = _ts(data.get(key))
        if t is not None and t >= _REAL_DATE:
            return t
    return None


def is_h3(data: dict[str, Any]) -> bool:
    text = " ".join(str(data.get(k) or "") for k in ("who", "what", "cmd"))
    return bool(_H3.search(text + " " + " ".join(map(str, data.get("argv") or []))))


def kind_of(who: str) -> str:
    """The --who with its numbers taken out: «a-x12-voices» and «a-x13-voices» are one kind."""
    return re.sub(r"\d+", "#", who.lower())


def kind_medians(home: Path | None, limit: int = 6000) -> dict[tuple[str, str], float]:
    """Median run time of finished runs by (lane, kind of --who), from history.jsonl."""
    if home is None:
        return {}
    path = home / "history.jsonl"
    try:
        st = path.stat()
    except OSError:
        return {}
    key = (st.st_mtime_ns, st.st_size)
    cached = _KIND_CACHE.get(path)
    if cached and cached[0] == key:
        return cached[1]
    runs: dict[tuple[str, str], list[float]] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-limit:]
    except OSError:
        lines = []
    for line in lines:
        try:
            r = json.loads(line)
            run = float(r.get("run_s") or 0)
        except (ValueError, TypeError, AttributeError):
            continue
        if run <= 0 or r.get("exit") not in (0, None) or r.get("state") == "waiting":
            continue
        runs.setdefault((str(r.get("lane")), kind_of(str(r.get("who", "")))), []).append(run)
    medians = {k: statistics.median(v) for k, v in runs.items()}
    _KIND_CACHE[path] = (key, medians)
    return medians


def expected_s(
    lane: str, data: dict[str, Any], medians: dict[tuple[str, str], float]
) -> float | None:
    """How long the run takes: the median of its kind in this lane, else the ticket's expect_s."""
    found = medians.get((lane, kind_of(str(data.get("who", "")))))
    if found is not None:
        return found
    try:
        return float(data["expect_s"])
    except (KeyError, TypeError, ValueError):
        return None


def tier(lane: str, data: dict[str, Any], medians: dict[tuple[str, str], float],
         now: float) -> tuple[int, float]:
    """(0, slot) for a near slot, (1, 0) for a short run, (2, 0) for the rest."""
    slot = slot_of(data)
    if slot is not None and slot - now < near_slot_s():
        return 0, slot
    expect = expected_s(lane, data, medians)
    if expect is not None and expect < short_s() and not is_h3(data):
        return 1, 0.0
    return 2, 0.0
