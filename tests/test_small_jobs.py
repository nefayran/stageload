"""Jobs predicted at most 1 GB start while a big render runs: they wait only for swap over the start
gate or the level under the pause floor.

2026-10-24: 85 jobs under 1 GB waited 159 min in all behind the shut gate; replayed, the rule
would have removed up to 77 min. The budget, the pause and kill rules and every bigger job stay
as they were.
"""

import json
import os
import sys

import pytest

from stageload import queue
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import is_small_job, run_queued

GB = 2**30
GATE = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35", "STAGELOAD_PAUSE_LEVEL": "25"}
TRUE = [sys.executable, "-c", "pass"]


@pytest.fixture
def gate(monkeypatch):
    for name, value in GATE.items():
        monkeypatch.setenv(name, value)
    for name in ("STAGELOAD_SMALL_JOB_GB", "STAGELOAD_REMOTE_LANES", "STAGELOAD_RESERVE_GB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def _io(logs, compressor=16 * GB, level=30, swap=4 * GB):
    """06:30Z with a big render running: the compressor 16 GB, the level 30 %, swap 4 GB."""
    return GuardIO(level=lambda: level, swap_used=lambda: swap, processes=lambda: [],
                   log=logs.append, compressor=lambda: compressor)


def _run(tmp_path, logs, io=None, **kw):
    kw.setdefault("who", "hq-lockfile")
    return run_queued(TRUE, GuardConfig(poll=0.05, start_timeout=0.4, wait_free=0),
                      io=io or _io(logs), home=tmp_path, trace=None, **kw)


def test_a_small_job_starts_past_the_compressor_and_a_level_over_the_floor(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs, lane="light", mem_peak=GB) == 0
    assert not any("waiting for memory" in line for line in logs)


def test_a_bigger_job_still_waits_at_the_start_gate(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs, lane="light", mem_peak=GB + 1) == 4
    wait = next(line for line in logs if "waiting for memory" in line)
    assert "compressor 16.0 GB > 8.0 GB" in wait and "start gate swap 6.0 GB" in wait


@pytest.mark.parametrize("swap, level, why", [
    (25 * GB, 30, "swap 25.0 GB > 24.0 GB"),
    (0, 20, "memory available 20% < 25%"),
])
def test_a_small_job_waits_for_swap_and_the_pause_floor(tmp_path, gate, swap, level, why):
    logs = []
    assert _run(tmp_path, logs, io=_io(logs, swap=swap, level=level), lane="light",
                mem_peak=GB // 2) == 4
    wait = next(line for line in logs if "waiting for memory" in line)
    assert why in wait and "small-job gate swap 24.0 GB, available 25%" in wait


def test_the_budget_still_holds_a_small_job(tmp_path, gate):
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    (lane / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps(
        {"pid": os.getpid(), "who": "box-c003", "lane": "ram", "state": "running",
         "mem_peak": 32 * GB}))
    logs = []
    assert _run(tmp_path, logs, lane="light", mem_peak=GB) == 4  # 32 + 1 + 16 > 48
    assert any("budget: running 32.0 GB" in line for line in logs)


def test_who_counts_as_small(gate, monkeypatch):
    small = {"mem_peak": GB, "lane": "light", "cmd": "python tools/thumb.py"}   # 26.10: codex is a remote lane now
    assert is_small_job(small)
    assert not is_small_job({**small, "mem_peak": 4 * GB})  # a new --who: 4 GB
    assert is_small_job({**small, "mem_peak": int(0.3 * GB * 1.3)})  # once history has it
    assert not is_small_job({**small, "gpu": True})
    assert not is_small_job({**small, "lane": "win"})
    for cmd in ("tools/Godot.app/Contents/MacOS/Godot --headless", "tools/film_look.sh a b",
                "zsh tools/movie.sh camp", "python tools/flow_check.py"):
        # history reads 0.01 GB for these: Godot opened through `open` holds ~6.9 GB
        assert not is_small_job({**small, "mem_peak": int(0.01 * GB), "cmd": cmd}), cmd
    assert not is_small_job({"lane": "light", "cmd": "x"})  # an old ticket without a peak
    monkeypatch.setenv("STAGELOAD_SMALL_JOB_GB", "2")
    assert is_small_job({**small, "mem_peak": 2 * GB})


def test_a_multi_lane_run_uses_the_small_job_gate_for_its_local_lane(tmp_path, gate):
    logs = []
    win = tmp_path / "queue" / "win"
    win.mkdir(parents=True)
    (win / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps(
        {"pid": os.getpid(), "who": "omb-cugan", "lane": "win", "state": "running"}))
    assert _run(tmp_path, logs, lane="win,light", mem_peak=GB // 2) == 0
    assert any("took the light lane of win,light" in line for line in logs)


def test_the_default_wait_free_does_not_hold_a_small_job(tmp_path, gate):
    # 30.10: --wait-free's default 35 % held a 0.8 GB job at 27 % although its own floor is 25 %
    logs = []
    config = GuardConfig(poll=0.05, start_timeout=0.4)  # wait_free left at its default
    assert run_queued(TRUE, config, io=_io(logs, swap=11 * GB, level=27), home=tmp_path, trace=None,
                      who="janitor-diskmap", lane="light", mem_peak=int(0.8 * GB)) == 0, logs
    logs = []
    assert run_queued(TRUE, config, io=_io(logs, swap=11 * GB, level=27), home=tmp_path, trace=None,
                      who="big-job", lane="light", mem_peak=4 * GB) == 4
