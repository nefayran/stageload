"""A --gpu head that waits for memory keeps its slot but runs nothing: a short job may use the
express place.

25.11: thorn-ep01-V07ap (--gpu) waited from 13:04Z for the 75 % memory gate under a 96-minute run
that held the other ram slot. Four jobs that fitted the budget (key checks, a sound bed) waited
90 minutes with «the lane's slots are taken by runs that passed while we waited». While such a
head waits, one non-GPU job expected to end within HEAD_BACKFILL_S takes the express place; the
gate and the budget still hold it and a long job still waits for a slot.
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


@pytest.fixture
def machine(monkeypatch):
    for name, value in {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
                        "STAGELOAD_GATE_LEVEL": "35"}.items():
        monkeypatch.setenv(name, value)
    for name in ("STAGELOAD_RESERVE_GB", "STAGELOAD_REMOTE_LANES", "STAGELOAD_SHARES"):
        monkeypatch.delenv(name, raising=False)
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


def _lane(home):
    """A long run and a --gpu head waiting for the memory gate hold both slots."""
    return [
        _ticket(home, "drama-ep02-voices", 1, state="running", started="2026-11-25T13:43:34+00:00",
                mem_peak=15 * GB),
        _ticket(home, "thorn-ep01-V07ap", 2, state="mem-wait", gpu=True, gpu_turn=True,
                mem_wait="memory available 54% < 75%", mem_peak=30 * GB),
    ]


def test_a_short_job_has_its_turn_while_a_gpu_head_waits_for_memory(tmp_path, machine):
    live = _lane(tmp_path) + [_ticket(tmp_path, "keys-find-w17", 3, state="waiting",
                                      mem_peak=10 * GB, expect_s=120)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2) == ([], "")


def test_a_long_job_still_waits_for_a_slot(tmp_path, machine):
    live = _lane(tmp_path) + [_ticket(tmp_path, "w16-bed", 3, state="waiting", mem_peak=4 * GB,
                                      expect_s=3600)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2)[1].startswith("2 ahead")


def test_no_place_is_lent_when_the_gpu_job_runs(tmp_path, machine):
    live = [_ticket(tmp_path, "drama-ep02-voices", 1, state="running", mem_peak=15 * GB),
            _ticket(tmp_path, "thorn-ep01-V07ap", 2, state="running", gpu=True, gpu_turn=True,
                    mem_peak=30 * GB),
            _ticket(tmp_path, "keys-find-w17", 3, state="waiting", mem_peak=10 * GB,
                    expect_s=120)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2)[1].startswith("2 ahead")


def test_it_starts_beside_two_held_slot_locks(tmp_path, machine):
    _history(tmp_path, "keys-find-w17", 120)
    _lane(tmp_path)
    lane_dir = tmp_path / "queue" / "ram"
    held = [(lane_dir / name).open("a") for name in ("run.lock", "run-1.lock")]
    for lock in held:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    io = GuardIO(level=lambda: 80, swap_used=lambda: 0, processes=lambda: [],
                 log=lambda line: None, compressor=lambda: GB)
    config = GuardConfig(poll=0.05, grace=1.0, wait_free=0, start_timeout=5)
    try:
        code = run_queued([sys.executable, "-c", "pass"], config, io=io, home=tmp_path,
                          who="keys-find-w17", mem_peak=10 * GB, trace=None)
    finally:
        for lock in held:
            lock.close()
    assert code == 0


def test_a_job_that_gave_its_slot_up_retakes_the_lent_place(tmp_path, machine):
    """It waited for memory with its slot released, the slots were taken meanwhile."""
    _history(tmp_path, "keys-find-w17", 120)
    _lane(tmp_path)
    lane_dir = tmp_path / "queue" / "ram"
    held = [(lane_dir / name).open("a") for name in ("run.lock", "run-1.lock")]
    for lock in held:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    level = [20]       # under the gate first: the job gives its slot up
    logs = []
    def log(line):
        logs.append(line)
        if "waiting for memory" in line:
            level[0] = 80
    io = GuardIO(level=lambda: level[0], swap_used=lambda: 0, processes=lambda: [],
                 log=log, compressor=lambda: GB)
    config = GuardConfig(poll=0.05, grace=1.0, wait_free=0, start_timeout=5)
    try:
        code = run_queued([sys.executable, "-c", "pass"], config, io=io, home=tmp_path,
                          who="keys-find-w17", mem_peak=10 * GB, trace=None)
    finally:
        for lock in held:
            lock.close()
    assert code == 0
