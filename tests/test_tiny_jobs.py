import sys

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
    for name in ("STAGELOAD_SMALL_JOB_GB", "STAGELOAD_TINY_JOB_GB",
                 "STAGELOAD_REMOTE_LANES", "STAGELOAD_RESERVE_GB"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)
    monkeypatch.setattr(queue, "disk_free", lambda home: 48 * GB)


def _run(tmp_path, logs, *, mem_peak, level=60, swap=12 * GB):
    # the compressor over 24 GB: swap is not stale here (29.10), so the small-job gate decides
    io = GuardIO(level=lambda: level, swap_used=lambda: swap, processes=[],
                 log=logs.append, compressor=lambda: 30 * GB)
    return run_queued(TRUE, GuardConfig(poll=0.05, start_timeout=0.4, wait_free=0),
                      io=io, home=tmp_path, trace=None, who="hq-lockfile", lane="light",
                      mem_peak=mem_peak)


def test_tiny_job_ignores_swap_above_start_gate(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs, mem_peak=int(0.2 * GB), level=60) == 0
    assert not any("waiting for memory" in line for line in logs)


def test_tiny_job_still_waits_below_pause_floor(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs, mem_peak=int(0.2 * GB), level=20) == 4
    wait = next(line for line in logs if "waiting for memory" in line)
    assert "memory available 20% < 25%" in wait
    assert "tiny-job gate: available 25% only (peak at most 0.2 GB)" in wait
    assert "swap 12.0 GB" not in wait


def test_larger_small_job_still_waits_on_swap(tmp_path, gate):
    logs = []
    assert _run(tmp_path, logs, mem_peak=int(0.8 * GB), swap=25 * GB) == 4
    wait = next(line for line in logs if "waiting for memory" in line)
    assert "swap 25.0 GB > 24.0 GB" in wait
    assert "small-job gate swap 24.0 GB" in wait


def test_zero_tiny_job_limit_disables_tiny_rule(tmp_path, gate, monkeypatch):
    monkeypatch.setenv("STAGELOAD_TINY_JOB_GB", "0")
    logs = []
    assert _run(tmp_path, logs, mem_peak=int(0.2 * GB), swap=25 * GB) == 4
    wait = next(line for line in logs if "waiting for memory" in line)
    assert "swap 25.0 GB > 24.0 GB" in wait
    assert "small-job gate swap 24.0 GB" in wait
