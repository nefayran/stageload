"""GPU starts cap --wait-free while retaining the memory gate and queue budget."""

import sys

import pytest

from stageload import queue
from stageload.guard import EXIT_NO_ROOM, GuardConfig, GuardIO
from stageload.queue import run_queued

GB = 2**30


@pytest.fixture
def machine(monkeypatch):
    for name, value in {
        "STAGELOAD_GATE_SWAP_GB": "6",
        "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35",
    }.items():
        monkeypatch.setenv(name, value)
    for name in (
        "STAGELOAD_GPU_WAIT_FREE_MAX",
        "STAGELOAD_RESERVE_GB",
        "STAGELOAD_REMOTE_LANES",
        "STAGELOAD_SHARES",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


@pytest.mark.parametrize(
    ("gpu", "level", "wait_free", "cap", "reason"),
    [
        (True, 50, 75, None, None),
        (True, 40, 75, None, "memory available 40% < 45%"),
        (False, 50, 75, None, "memory available 50% < 75%"),
        (True, 50, 75, "60", "memory available 50% < 60%"),
        (True, 40, 35, None, None),
        (True, 50, 75, "90", "memory available 50% < 75%"),
    ],
)
def test_wait_free_cap(tmp_path, machine, monkeypatch, gpu, level, wait_free, cap, reason):
    if cap is not None:
        monkeypatch.setenv("STAGELOAD_GPU_WAIT_FREE_MAX", cap)
    logs = []
    io = GuardIO(level=lambda: level, swap_used=lambda: 0, processes=lambda: [],
                 log=logs.append, compressor=lambda: GB)
    config = GuardConfig(poll=0.05, grace=1.0, wait_free=wait_free, start_timeout=0.2)
    started = tmp_path / "started"
    cmd = [sys.executable, "-c", "import pathlib, sys; pathlib.Path(sys.argv[1]).touch()", str(started)]

    code = run_queued(cmd, config, io=io, home=tmp_path, who="thorn-ep01-V07ap",
                      gpu=gpu, mem_peak=25 * GB, trace=None)

    assert config.wait_free == wait_free
    if reason is None:
        assert code == 0, logs
        assert started.exists()
        assert not any("waiting for memory" in line for line in logs)
    else:
        assert code == EXIT_NO_ROOM, logs
        assert not started.exists()
        assert any(f"waiting for memory ({reason};" in line for line in logs), logs
