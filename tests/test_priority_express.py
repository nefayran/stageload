"""A priority run of seconds takes the lane's express place when its slots are held.

A 15-second job its priority line asked to run first waited an hour and more: the GPU was free,
but both ram slots were held by CPU jobs of 85 minutes and two hours, and three runs waiting for
memory went before it again after twenty minutes. Such a run (a priority place, expected under
STAGELOAD_PRIORITY_TINY_S) now starts beside the slots, one at a time. The start gate, the
budget and the GPU still hold it, and the express run takes no slot from the runs behind it.
"""

import fcntl
import json
import os
import sys
import time
from datetime import datetime, timezone

import pytest

from stageload import queue
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import lane_ahead, run_queued

GB = 2**30
PLACE = "1900-01-01T00:00Z"   # a priority line's place, as a cleaner writes it


@pytest.fixture
def machine(monkeypatch):
    for name, value in {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
                        "STAGELOAD_GATE_LEVEL": "35"}.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STAGELOAD_RESERVE_GB", raising=False)
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    monkeypatch.delenv("STAGELOAD_SHARES", raising=False)
    monkeypatch.setenv("STAGELOAD_RAM_SLOTS", "2")
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def _history(home, who, run_s):
    end = datetime.now(timezone.utc).isoformat()
    with open(home / "history.jsonl", "a") as f:
        for _ in range(5):
            f.write(json.dumps({"lane": "ram", "who": who, "run_s": run_s, "end": end,
                                "exit": 0}) + "\n")


def _ticket(home, who, n, **data):
    lane_dir = home / "queue" / "ram"
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / f"{n:020d}-{os.getpid()}.json"
    entry = {"pid": os.getpid(), "who": who, "lane": "ram", "cmd": "python job.py", **data}
    path.write_text(json.dumps(entry))
    return path, entry


def _held_lane(home):
    """Two long CPU runs hold both slots; a run waiting for memory yielded over 20 minutes."""
    long_ago = time.time() - 3600
    return [
        _ticket(home, "now-sim-chunk", 1, state="running", started="2026-11-18T12:32:12+00:00",
                mem_peak=4 * GB),
        _ticket(home, "keys-find-w14", 2, state="running", started="2026-11-18T12:54:47+00:00",
                mem_peak=8 * GB),
        _ticket(home, "w17-vo-cast", 3, state="mem-wait", blocked="the lane's slots are taken",
                blocked_at=time.time(), blocked_since=long_ago, released=True),
    ]


def test_a_tiny_priority_run_has_its_turn_while_the_slots_are_held(tmp_path, machine):
    _history(tmp_path, "narration-voice", 15)
    live = _held_lane(tmp_path) + [_ticket(tmp_path, "narration-voice", 4, state="waiting",
                                           gpu=True, deadline=PLACE)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2) == ([], "")


def test_a_run_of_minutes_on_the_same_line_still_waits_for_a_slot(tmp_path, machine):
    _history(tmp_path, "narration-batch", 600)
    live = _held_lane(tmp_path) + [_ticket(tmp_path, "narration-batch", 4, state="waiting",
                                           deadline=PLACE)]
    ahead, reason = lane_ahead(live[-1][0], live, {}, None, slots=2)
    assert reason.startswith("3 ahead")


def test_the_express_run_takes_no_slot_from_the_runs_behind_it(tmp_path, machine):
    live = [_ticket(tmp_path, "now-sim-chunk", 1, state="running"),
            _ticket(tmp_path, "narration-voice", 2, state="running", express=True),
            _ticket(tmp_path, "layout", 3, state="waiting")]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2) == ([], "")


def test_it_starts_beside_two_held_slot_locks(tmp_path, machine):
    _history(tmp_path, "narration-voice", 15)
    _held_lane(tmp_path)
    lane_dir = tmp_path / "queue" / "ram"
    held = [(lane_dir / name).open("a") for name in ("run.lock", "run-1.lock")]
    for lock in held:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    io = GuardIO(level=lambda: 80, swap_used=lambda: 0, processes=lambda: [],
                 log=lambda line: None, compressor=lambda: GB)
    config = GuardConfig(poll=0.05, grace=1.0, wait_free=0, start_timeout=5)
    try:
        code = run_queued([sys.executable, "-c", "pass"], config, io=io, home=tmp_path,
                          who="narration-voice", deadline=PLACE, mem_peak=8 * GB, trace=None)
    finally:
        for lock in held:
            lock.close()
    assert code == 0
