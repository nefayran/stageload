"""A run may wait in several lanes and take the first that can start it; the ram lane runs a
second job beside the first when the budget fits.

2026-10-24: two Whisper checks waited for hours behind a CUGAN job in a remote lane while the
Mac's light lane stood empty.
"""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from stageload import queue
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import _without_wrappers, lane_ahead, list_queue, run_queued

GB = 2**30
GATE = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35"}


@pytest.fixture
def machine(monkeypatch):
    for name, value in GATE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STAGELOAD_RESERVE_GB", raising=False)
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    monkeypatch.setenv("STAGELOAD_RAM_SLOTS", "2")
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


class Readings:
    def __init__(self, home, compressor=GB):
        self.home, self.compressor = home, compressor
        self.logs, self.seen = [], []

    def io(self):
        def swap():
            self.seen.append(list_queue(self.home))
            return 0

        return GuardIO(level=lambda: 80, swap_used=swap, processes=lambda: [],
                       log=self.logs.append, compressor=lambda: self.compressor)


def _lane_writer(path):
    code = f"import os, pathlib; pathlib.Path({str(path)!r}).write_text(os.environ['STAGELOAD_LANE'])"
    return [sys.executable, "-c", code]


def _holder(home, lane, who, **data):
    """A run of another session that holds the lane: its pid is this test process, so it lives."""
    lane_dir = home / "queue" / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    ticket = lane_dir / f"{time.time_ns():020d}-{os.getpid()}.json"
    ticket.write_text(json.dumps({"pid": os.getpid(), "who": who, "lane": lane,
                                  "cmd": "python job.py", "state": "running",
                                  "started": "2026-10-24T06:00:00+00:00", **data}))
    return ticket


def _quick(**fields):
    return GuardConfig(poll=0.05, grace=1.0, wait_free=0, **fields)


def _later(seconds, action):
    timer = threading.Timer(seconds, action)
    timer.start()
    return timer


# --- several lanes -------------------------------------------------------------------------


def test_a_run_of_two_lanes_takes_the_free_one_and_drops_the_other(tmp_path, machine):
    _holder(tmp_path, "win", "omb-cugan-4k", mem_peak=0)
    out = tmp_path / "lane.txt"
    r = Readings(tmp_path)
    assert run_queued(_lane_writer(out), _quick(), io=r.io(), home=tmp_path, lane="win,light",
                      who="w08-whisper-check", trace=None) == 0
    assert out.read_text() == "light"
    assert any("took the light lane of win,light" in line for line in r.logs)
    # once it runs, it is in light alone: its entry in win is gone, the holder's stays
    assert [(x["lane"], x["state"]) for x in r.seen[-1] if x["who"] == "w08-whisper-check"] == [
        ("light", "running")]
    assert len(list((tmp_path / "queue" / "win").glob("*.json"))) == 1
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["lane"] == "light"


def test_the_first_listed_lane_wins_when_both_are_free_and_a_remote_one_skips_the_gate(
        tmp_path, machine):
    out = tmp_path / "lane.txt"
    r = Readings(tmp_path, compressor=20 * GB)  # this Mac is over the gate: light cannot start
    assert run_queued(_lane_writer(out), _quick(), io=r.io(), home=tmp_path, lane="win,light",
                      who="w08-whisper-check", trace=None) == 0
    assert out.read_text() == "win"


def test_a_run_waits_in_each_lane_until_one_can_start_it(tmp_path, machine):
    holder = _holder(tmp_path, "win", "omb-cugan-4k", mem_peak=0)
    out = tmp_path / "lane.txt"
    # light: over the gate (and over the 24 GB up to which the compressor is stale at 80 %);
    # win: taken
    r = Readings(tmp_path, compressor=30 * GB)
    timer = _later(0.5, lambda: holder.unlink())
    try:
        assert run_queued(_lane_writer(out), _quick(), io=r.io(), home=tmp_path,
                          lane="win,light", who="w08-whisper-check", trace=None) == 0
    finally:
        timer.cancel()
    assert out.read_text() == "win"
    wait = next(line for line in r.logs if "waiting in win,light" in line)
    assert "win: 1 ahead: omb-cugan-4k" in wait and "light: compressor 30.0 GB > 8.0 GB" in wait


def test_a_waiting_run_of_two_lanes_shows_in_both(tmp_path, machine):
    holder = _holder(tmp_path, "win", "omb-cugan-4k", mem_peak=0)
    _holder(tmp_path, "light", "omb-build", mem_peak=0)
    r = Readings(tmp_path)
    shown = []

    def look():
        shown.extend((x["lane"], x["state"]) for x in list_queue(tmp_path)
                     if x["who"] == "w08-whisper-check")
        holder.unlink()

    timer = _later(0.4, look)
    try:
        assert run_queued([sys.executable, "-c", "pass"], _quick(), io=r.io(), home=tmp_path,
                          lane="win,light", who="w08-whisper-check", trace=None) == 0
    finally:
        timer.cancel()
    assert sorted(shown) == [("light", "waiting"), ("win", "waiting")]


def test_a_single_lane_run_gets_its_lane_too(tmp_path, machine):
    out = tmp_path / "lane.txt"
    assert run_queued(_lane_writer(out), _quick(), io=Readings(tmp_path).io(), home=tmp_path,
                      lane="light", who="x", trace=None) == 0
    assert out.read_text() == "light"


# --- a second ram run ----------------------------------------------------------------------


def test_a_second_ram_run_starts_beside_the_first_when_the_budget_fits(tmp_path, machine):
    _holder(tmp_path, "ram", "w09-draft", mem_peak=16 * GB)  # 16 + ours 16 + reserve 16 = 48
    r = Readings(tmp_path)
    assert run_queued([sys.executable, "-c", "pass"], _quick(start_timeout=3), io=r.io(),
                      home=tmp_path, who="w10-draft", trace=None) == 0
    assert not any("waiting" in line for line in r.logs), r.logs


def test_a_second_ram_run_waits_while_the_budget_does_not_fit(tmp_path, machine):
    holder = _holder(tmp_path, "ram", "code-isekai-omb-c104bLv4", mem_peak=36 * GB, gpu=True)
    r = Readings(tmp_path)
    timer = _later(0.5, lambda: holder.unlink())
    try:
        assert run_queued([sys.executable, "-c", "pass"], _quick(), io=r.io(), home=tmp_path,
                          who="w10-draft", trace=None) == 0
    finally:
        timer.cancel()
    wait = next(line for line in r.logs if "waiting for memory" in line)
    assert "budget: running 36.0 GB (code-isekai-omb-c104bLv4 (ram) 36 GB)" in wait


def test_an_undeclared_running_job_counts_at_its_predicted_peak(tmp_path, machine):
    # the morning's crash: a render that declared nothing read 6.43 GB and held some 25
    holder = _holder(tmp_path, "ram", "box-c003",
                     cmd="zsh /Users/me/channel2-growth/tools/h3_exec_convrot.sh --profile")
    r = Readings(tmp_path)
    timer = _later(0.5, lambda: holder.unlink())
    try:
        assert run_queued([sys.executable, "-c", "pass"], _quick(), io=r.io(), home=tmp_path,
                          who="w10-draft", trace=None) == 0
    finally:
        timer.cancel()
    assert any("budget: running 25.0 GB (box-c003 (ram) 25 GB)" in line for line in r.logs)


def test_two_gpu_runs_never_run_together_in_the_ram_lane(tmp_path, machine):
    holder = _holder(tmp_path, "ram", "box-c003", mem_peak=4 * GB, gpu=True, gpu_turn=True)
    r = Readings(tmp_path)
    timer = _later(0.5, lambda: holder.unlink())
    try:
        assert run_queued([sys.executable, "-c", "pass"], _quick(), io=r.io(), home=tmp_path,
                          who="sh56-c5", gpu=True, mem_peak=4 * GB, trace=None) == 0
    finally:
        timer.cancel()
    # it took no slot while box-c003 held the GPU: the second slot stays for jobs without --gpu
    assert any("waiting in ram (the GPU is taken, 1 ahead: box-c003 (ram))" in line
               for line in r.logs)


def test_the_second_slot_keeps_the_lanes_order():
    run = (Path("1-1.json"), {"who": "h3", "state": "running"})
    old = (Path("2-1.json"), {"who": "older", "state": "waiting"})
    new = (Path("3-1.json"), {"who": "newer", "state": "waiting"})
    live = [run, old, new]
    assert lane_ahead(new[0], live, {}, None, slots=2)[1] == "2 ahead: h3, older"
    assert lane_ahead(old[0], live, {}, None, slots=2) == ([], "")
    assert lane_ahead(old[0], live, {}, None, slots=1)[1] == "1 ahead: h3"
    both = [run, (Path("4-1.json"), {"who": "w09", "state": "mem-wait"}), new]
    assert lane_ahead(new[0], both, {}, None, slots=2)[1] == "2 ahead: h3, w09"


def test_an_engine_a_running_run_started_does_not_hold_the_second_one_as_busy(tmp_path):
    _holder(tmp_path, "ram", "box-c003")
    me = os.getpid()
    rows = [(me, 1, "python -m stageload.queue spec.json"),
            (5001, me, "/Users/me/local-llm/h3-forks/lachlanstuart/h3 --profile"),
            (6001, 1, "/Users/me/local-llm/h3-forks/lachlanstuart/h3 --outside")]
    listing = _without_wrappers(lambda: rows, tmp_path)()
    assert [pid for pid, _, _ in listing] == [6001]


def test_two_real_ram_runs_overlap_in_the_two_slots(tmp_path, machine):
    marks = tmp_path / "marks"
    marks.mkdir()

    def job(name):
        return [sys.executable, "-c",
                f"import pathlib, time; d = pathlib.Path({str(marks)!r}); "
                f"(d / '{name}.start').write_text(str(time.time())); time.sleep(1.0); "
                f"(d / '{name}.end').write_text(str(time.time()))"]

    codes = {}

    def run(name):
        codes[name] = run_queued(job(name), _quick(), io=Readings(tmp_path).io(), home=tmp_path,
                                 who=f"w1{name}-draft", trace=None)

    threads = [threading.Thread(target=run, args=(name,)) for name in ("a", "b")]
    for thread in threads:
        thread.start()
        time.sleep(0.2)
    for thread in threads:
        thread.join(timeout=20)
    assert codes == {"a": 0, "b": 0}
    start_b = float((marks / "b.start").read_text())
    assert start_b < float((marks / "a.end").read_text())  # b started while a still ran
    assert {p.name for p in (tmp_path / "queue" / "ram").glob("run*.lock")} == {
        "run.lock", "run-1.lock"}


def test_codex_lane_holds_three_runs_and_any_lane_reads_its_own_slot_count(monkeypatch):
    # jobs in the codex lane call a hosted image service, so three may run at once by default
    monkeypatch.delenv("STAGELOAD_CODEX_SLOTS", raising=False)
    monkeypatch.delenv("STAGELOAD_WIN_SLOTS", raising=False)
    assert queue.lane_slots("codex") == 3
    assert queue.lane_slots("win") == 1 and queue.lane_slots("arch") == 1
    monkeypatch.setenv("STAGELOAD_WIN_SLOTS", "1")
    assert queue.lane_slots("win") == 1
    monkeypatch.setenv("STAGELOAD_CODEX_SLOTS", "not a number")
    assert queue.lane_slots("codex") == 3
