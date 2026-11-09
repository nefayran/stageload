"""Swap over the start gate does not hold a start when the Mac is idle: no local queue job runs,
the level is at the gate or over it and the compressor is under its gate.

Up to 3.9 h a day of start time was lost to stale swap with no big render running. The pause and kill
rules are unchanged.
"""

import json
import os
import sys
import time

import pytest

from stageload import queue
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import run_queued

GB = 2**30
GATE = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35", "STAGELOAD_PAUSE_LEVEL": "25"}
TRUE = [sys.executable, "-c", "pass"]


@pytest.fixture
def gate(monkeypatch):
    for name, value in GATE.items():
        monkeypatch.setenv(name, value)
    for name in ("STAGELOAD_REMOTE_LANES", "STAGELOAD_RESERVE_GB", "STAGELOAD_SMALL_JOB_GB",
                 "STAGELOAD_GATE_STALE_SWAP_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def _run(tmp_path, logs, swap=int(9.4 * GB), compressor=5 * GB, level=60, **kw):
    io = GuardIO(level=lambda: level, swap_used=lambda: swap, processes=lambda: [],
                 log=logs.append, compressor=lambda: compressor)
    return run_queued(TRUE, GuardConfig(poll=0.05, start_timeout=0.4, wait_free=0), io=io,
                      home=tmp_path, trace=None, who="w09-draft", mem_peak=4 * GB, **kw)


def _running(home, lane, who="box-c003"):
    lane_dir = home / "queue" / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    (lane_dir / f"{time.time_ns():020d}-{os.getpid()}.json").write_text(json.dumps(
        {"pid": os.getpid(), "who": who, "lane": lane, "state": "running", "mem_peak": 0}))


def test_stale_swap_on_an_idle_mac_does_not_hold_the_start(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs) == 0
    note = next(line for line in logs if "starting although" in line)
    assert "swap 9.4 GB > 6.0 GB, but no local queue job runs" in note and "stale" in note


def test_a_remote_job_does_not_make_the_mac_busy(tmp_path, gate):
    _running(tmp_path, "win", "omb-cugan")
    logs = []
    assert _run(tmp_path, logs) == 0


@pytest.mark.parametrize("why, kw", [
    ("a local queue job runs and the level is under the 35 % stale level", {"level": 34}),
    ("the level is under the gate", {"level": 30}),
    ("the compressor is over its stale ceiling", {"compressor": 25 * GB}),
])
def test_swap_still_holds_the_start_otherwise(tmp_path, gate, why, kw):
    if why.startswith("a local queue job runs"):
        _running(tmp_path, "light")
    logs = []
    assert _run(tmp_path, logs, **kw) == 4, why
    wait = next(line for line in logs if "waiting for memory" in line)
    assert "swap 9.4 GB > 6.0 GB" in wait


def test_a_small_job_on_an_idle_mac_is_let_through_too(tmp_path, gate):
    logs = []
    io = GuardIO(level=lambda: 60, swap_used=lambda: 9 * GB, processes=lambda: [],
                 log=logs.append, compressor=lambda: 2 * GB)
    assert run_queued(TRUE, GuardConfig(poll=0.05, start_timeout=0.4, wait_free=0), io=io,
                      home=tmp_path, trace=None, who="hq-lockfile", lane="light",
                      mem_peak=GB // 2) == 0


def _run_limited(tmp_path, logs, swap=int(7.6 * GB), compressor=int(1.3 * GB), level=90):
    io = GuardIO(level=lambda: level, swap_used=lambda: swap, processes=lambda: [],
                 log=logs.append, compressor=lambda: compressor)
    config = GuardConfig(poll=0.05, start_timeout=0.4, wait_free=0, swap_limit=6 * GB, swap_budget=3 * GB)
    return run_queued(TRUE, config, io=io, home=tmp_path, trace=None, who="now-chunk", mem_peak=GB)


def test_a_runs_own_swap_limit_does_not_hold_on_an_idle_mac(tmp_path, gate):
    # 2026-10-26: a long job's chunk (--swap-limit 6G --swap-budget 3G) waited on «swap 7.6 GB + budget 3.0 GB > limit 6.0 GB» with
    # the Mac idle (90 % free, compressor 1.3 GB), 67 jobs behind it: the idle rule now covers that check too
    logs = []
    assert _run_limited(tmp_path, logs) == 0
    assert not any("swap 7.6 GB + budget" in line for line in logs)


def test_a_runs_own_swap_limit_still_holds_when_a_job_runs(tmp_path, gate):
    _running(tmp_path, "light")
    logs = []
    assert _run_limited(tmp_path, logs, level=34) == 4
    assert any("swap 7.6 GB + budget 3.0 GB > limit 6.0 GB" in line for line in logs)


def test_a_full_disk_holds_a_new_start(tmp_path, gate, monkeypatch):
    # 26.10: no new job starts on this Mac while its data volume has under STAGELOAD_GATE_DISK_GB free
    monkeypatch.setattr(queue, "disk_free", lambda home: int(7.4 * GB))
    logs = []
    assert _run(tmp_path, logs, swap=0) == 4
    assert any("disk: 7.4 GiB free < 8 GiB" in line for line in logs)


def test_enough_disk_lets_it_start(tmp_path, gate, monkeypatch):
    monkeypatch.setattr(queue, "disk_free", lambda home: 89 * GB)
    logs = []
    assert _run(tmp_path, logs, swap=0) == 0


# 2026-10-27: macOS does not page swap back in. At 07:33 UTC a video job waited on
# «swap 8.5 GB > 6.0 GB» at 84-88 % available with the compressor at 1.2 GB, because a 1 GB image job ran in light
def test_k17_stale_swap_with_plenty_of_memory_starts_next_to_a_running_job(tmp_path, gate):
    _running(tmp_path, "light", "manga-ch06-redo1")
    logs = []
    assert _run(tmp_path, logs, swap=int(8.5 * GB), compressor=int(1.2 * GB), level=85) == 0
    note = next(line for line in logs if "starting although" in line)
    assert "swap 8.5 GB > 6.0 GB, but memory available is 85% (>= 35%)" in note and "stale" in note


def test_k17_low_level_still_waits_next_to_a_running_job(tmp_path, gate):
    _running(tmp_path, "light", "manga-ch06-redo1")
    logs = []
    assert _run(tmp_path, logs, swap=int(8.5 * GB), compressor=int(1.2 * GB), level=34) == 4
    assert any("waiting for memory" in line and "swap 8.5 GB > 6.0 GB" in line for line in logs)


def test_k17_compressor_over_its_gate_still_waits(tmp_path, gate):
    # 29.10: a compressor up to 24 GB at 35 % or more is stale too; over that it still holds
    _running(tmp_path, "light")
    logs = []
    assert _run(tmp_path, logs, swap=int(8.5 * GB), compressor=30 * GB, level=85) == 4


def test_k17_covers_a_runs_own_swap_limit(tmp_path, gate):
    _running(tmp_path, "light")
    logs = []
    assert _run_limited(tmp_path, logs, level=85) == 0
    assert not any("swap 7.6 GB + budget" in line for line in logs)


def test_k17_floor_is_an_env_knob(tmp_path, gate, monkeypatch):
    monkeypatch.setenv("STAGELOAD_GATE_STALE_SWAP_LEVEL", "90")
    _running(tmp_path, "light")
    logs = []
    assert _run(tmp_path, logs, swap=int(8.5 * GB), compressor=int(1.2 * GB), level=85) == 4
