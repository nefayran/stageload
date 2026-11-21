"""`stageload plan`: put waiting runs in a chosen order, or mark the lane a waiting run should take.

A planner (a person, or a program that places work) names waiting runs by --who or --owner and gives
them a place: ``--first`` or ``--place N``. The place is written into the run's entry as a deadline
before 2002, the queue's form for a place (order.py), with ``planned_by`` and ``planned_why``; the
waiting run keeps these fields when it rewrites its entry (queue.KEPT_FIELDS). ``--clear`` takes a
place off again. ``--prefer-lane`` writes the lane the run should take when more than one can run it
(a waiting H3 clip on Windows rather than the Mac); the program that moves such runs reads it.

What the command never does:
- touch a run that has started (running or paused) or holds its lane's turn;
- change the start gates, the memory budget or the lanes' places: a planned run still starts only
  when its lane's gates and the machine's budget let it;
- give an H3 video clip a place unless ``allow_h3`` is set, because places on H3 are the owner's
  call (the office's rule), not a planner's.
"""

from __future__ import annotations

import fnmatch
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from . import order as order_rules
from .queue import _HOLDS_LANE, _TICKET, _all_tickets, _owner

# 21.11: place 0 was 2001-01-01, after the priority lines (1900 + index) and the planner's tiers
# (2000-*), so `--first` put a post's narration behind 40 clips; a planned place goes before both
PLACE_ZERO = datetime(1899, 1, 1, tzinfo=timezone.utc)
PLAN_FIELDS = ("deadline", "planned_by", "planned_why", "prefer_lane")


def place_deadline(place: int) -> str:
    """A place as the queue reads it: a deadline before 2002, earlier = sooner."""
    return (PLACE_ZERO + timedelta(seconds=max(int(place), 0))).isoformat().replace("+00:00", "Z")


def _write(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".tmp-plan-{path.name}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def plan(
    home: Path,
    *,
    who: str | None = None,
    owner: str | None = None,
    lane: str | None = None,
    place: int | None = None,
    clear: bool = False,
    prefer_lane: str | None = None,
    by: str = "planner",
    why: str = "",
    allow_h3: bool = False,
    dry: bool = False,
) -> list[dict[str, Any]]:
    """Apply a place or a lane preference to the waiting runs named; one row per entry."""
    if who is None and owner is None:
        raise ValueError("name the runs with who or owner")
    if place is None and not clear and prefer_lane is None:
        raise ValueError("give a place, clear, or a preferred lane")
    rows = []
    for path, data in _all_tickets(home):
        if not _TICKET.match(path.name):
            continue
        if lane is not None and path.parent.name != lane:
            continue
        name = str(data.get("who") or "")
        if who is not None and not fnmatch.fnmatchcase(name, who):
            continue
        if owner is not None and _owner(data) != owner:
            continue
        row = {
            "lane": path.parent.name,
            "who": name,
            "state": data.get("state", "waiting"),
            "done": "",
            "why": "",
        }
        if data.get("state") in _HOLDS_LANE:
            row.update(done="skipped", why="it has started or holds its lane's turn")
        elif place is not None and order_rules.is_h3(data) and not allow_h3:
            row.update(
                done="skipped",
                why="an H3 clip's place is the owner's call (pass allow_h3 with his word)",
            )
        else:
            new = dict(data)
            if clear:
                for key in PLAN_FIELDS:
                    new.pop(key, None)
                row["done"] = "cleared"
            if place is not None:
                new.update(deadline=place_deadline(place), planned_by=by, planned_why=why)
                row["done"] = f"place {place}"
            if prefer_lane is not None:
                new["prefer_lane"] = prefer_lane
                new.setdefault("planned_by", by)
                if why:
                    new["planned_why"] = why
                row["done"] = (row["done"] + ", " if row["done"] else "") + f"prefers {prefer_lane}"
            if not dry:
                _write(path, new)
        rows.append(row)
    return rows
