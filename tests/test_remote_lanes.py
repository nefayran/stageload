"""24.10: lanes that run on another machine are not held by this Mac's memory gate."""
from stageload.queue import remote_lanes


def test_remote_lanes_default(monkeypatch):
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    assert remote_lanes() == {"win", "win-cpu", "arch", "arch-gpu", "modal", "codex", "giz", "gcp"}


def test_a_modal_run_is_remote_by_default(tmp_path, monkeypatch):
    # KAIZEN k8: Pixal3D jobs run on Modal through `--lane modal` and use almost nothing here,
    # so this Mac's shut gate and budget do not hold them and its readings never pause them
    import sys

    from stageload.guard import GuardConfig, GuardIO
    from stageload.queue import run_queued

    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    for name, value in (("STAGELOAD_GATE_SWAP_GB", "6"), ("STAGELOAD_GATE_COMPRESSOR_GB", "8"),
                        ("STAGELOAD_GATE_LEVEL", "35")):
        monkeypatch.setenv(name, value)
    logs = []
    io = GuardIO(level=lambda: 20, swap_used=lambda: 12 * 2**30, processes=lambda: [],
                 log=logs.append, compressor=lambda: 26 * 2**30)
    code = run_queued([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, start_timeout=0.5,
                      wait_free=0), io=io, home=tmp_path, lane="modal", who="pixal3d-modal",
                      trace=None)
    assert code == 0
    assert not any("waiting for memory" in line for line in logs)
    assert any("never paused or stopped" in line for line in logs)


def test_remote_lanes_env(monkeypatch):
    monkeypatch.setenv("STAGELOAD_REMOTE_LANES", " win , gpu-box ,")
    assert remote_lanes() == {"win", "gpu-box"}


def test_a_remote_run_does_not_wait_for_this_macs_free_memory(tmp_path, monkeypatch):
    # 25.10 14:45 UTC: a win job and a modal job waited 30 min for this Mac's wait_free (40 %) and swap limit
    # in wait_for_room, which ran for every lane; a remote lane uses none of this Mac's memory
    import sys

    from stageload.guard import GuardConfig, GuardIO
    from stageload.queue import run_queued

    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    logs = []
    io = GuardIO(level=lambda: 20, swap_used=lambda: 12 * 2**30, processes=lambda: [],
                 log=logs.append, compressor=lambda: 26 * 2**30)
    for lane in ("win", "modal"):
        code = run_queued([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, start_timeout=0.5,
                          wait_free=40), io=io, home=tmp_path, lane=lane, who=f"{lane}-job", trace=None)
        assert code == 0, (lane, logs)
    assert not any("guard: waiting" in line or "no room" in line for line in logs)


def test_a_local_run_still_waits_for_free_memory(tmp_path, monkeypatch):
    import sys

    from stageload.guard import GuardConfig, GuardIO
    from stageload.queue import run_queued

    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    logs = []
    io = GuardIO(level=lambda: 20, swap_used=lambda: 0, processes=lambda: [], log=logs.append,
                 compressor=lambda: 0)
    code = run_queued([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, start_timeout=0.3,
                      wait_free=40), io=io, home=tmp_path, lane="light", who="light-job", trace=None)
    assert code != 0
    # 29.10: the room checks (--wait-free) are part of the memory wait, which gives the slot up
    assert any("waiting for memory" in line and "memory available 20% < 40%" in line
               for line in logs)
