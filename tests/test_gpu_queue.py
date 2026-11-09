"""GPU jobs take turns across lanes; a run that fails at once keeps its lane's next turn for its
owner.

2026-10-21: a GPU job of the light lane beside a GPU video render made the render 2.4 times slower.
2026-10-29: a GPU run waiting for the GPU held the second ram slot for a whole render, so a GPU
run takes no slot while the GPU is taken, and the jobs without --gpu behind it go first.
"""

import json
import subprocess
import sys
import time

from stageload.cli import _parser
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import (
    HOLD_S,
    QUICK_FAIL_HOLD_S,
    _hold_seconds,
    _quick_failure,
    gpu_ahead,
    lane_ahead,
    list_queue,
    run_queued,
)

RUN = """
import sys
from pathlib import Path
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import run_queued
io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [],
             log=lambda line: print(line, file=sys.stderr, flush=True))
sys.exit(run_queued({cmd!r}, GuardConfig(poll=0.05, grace=1.0), io=io, home=Path({home!r}),
                    lane={lane!r}, who={who!r}, trace=None, gpu={gpu!r}))
"""


def _job(log, who, seconds):
    code = (
        "import pathlib, sys, time; p = pathlib.Path(sys.argv[1]); "
        "f = p.open('a'); f.write(f'start {sys.argv[2]} {time.time()}\\n'); f.flush(); "
        f"time.sleep({seconds}); f.write(f'end {{sys.argv[2]}} {{time.time()}}\\n'); f.close()"
    )
    return [sys.executable, "-c", code, str(log), who]


def _start(home, log, who, seconds, lane, gpu):
    code = RUN.format(cmd=_job(log, who, seconds), home=str(home), lane=lane, who=who, gpu=gpu)
    return subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE, text=True)


def _times(log):
    out = {}
    for line in log.read_text().splitlines():
        kind, who, t = line.split()
        out[(kind, who)] = float(t)
    return out


def _wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never came true")


# --- B ------------------------------------------------------------------------------------


def test_gpu_jobs_of_two_lanes_never_overlap_and_others_still_run_beside(tmp_path):
    log = tmp_path / "log"
    tts = _start(tmp_path, log, "tts", 1.5, "light", True)
    _wait_for(lambda: log.exists() and "start tts" in log.read_text())
    render = _start(tmp_path, log, "render", 0.1, "ram", True)
    _wait_for(lambda: _held_by_gpu(tmp_path, "render"))
    layout = _start(tmp_path, log, "layout", 0.1, "layout", False)
    for run in (tts, render, layout):
        assert run.wait(timeout=30) == 0
    times = _times(log)
    assert times[("end", "tts")] <= times[("start", "render")]
    assert times[("start", "layout")] < times[("end", "tts")]  # a job without --gpu does not wait
    assert "the GPU is taken, 1 ahead: tts (light)" in render.stderr.read()


def _held_by_gpu(home, who):
    """The run's ticket says it has its lane's turn but the GPU is taken: it holds no slot."""
    for path in (home / "queue").glob("*/0*.json"):
        data = json.loads(path.read_text())
        if data.get("who") == who:
            return data.get("state") == "waiting" and bool(data.get("blocked_gpu"))
    return False


def test_a_job_without_gpu_behind_the_gpu_waiter_starts_first(tmp_path):
    # the GPU waiter keeps no slot while the GPU is taken, so blender, behind it in ram, runs
    # while tts holds the GPU; render takes the GPU after tts, in its turn
    log = tmp_path / "log"
    tts = _start(tmp_path, log, "tts", 1.5, "light", True)
    _wait_for(lambda: log.exists() and "start tts" in log.read_text())
    render = _start(tmp_path, log, "render", 0.1, "ram", True)
    _wait_for(lambda: _held_by_gpu(tmp_path, "render"))
    blender = _start(tmp_path, log, "blender", 0.1, "ram", False)
    for run in (tts, render, blender):
        assert run.wait(timeout=30) == 0
    times = _times(log)
    assert times[("start", "blender")] < times[("end", "tts")]
    assert times[("end", "tts")] <= times[("start", "render")]
    rows = list_queue(tmp_path)
    assert rows == []


def test_two_gpu_runs_of_one_lane_take_the_gpu_in_order(tmp_path):
    log = tmp_path / "log"
    tts = _start(tmp_path, log, "tts", 1.0, "light", True)
    _wait_for(lambda: log.exists() and "start tts" in log.read_text())
    first = _start(tmp_path, log, "first", 0.5, "ram", True)
    _wait_for(lambda: _held_by_gpu(tmp_path, "first"))
    second = _start(tmp_path, log, "second", 0.1, "ram", True)
    for run in (tts, first, second):
        assert run.wait(timeout=30) == 0
    times = _times(log)
    assert times[("end", "tts")] <= times[("start", "first")]
    assert times[("end", "first")] <= times[("start", "second")]


def test_gpu_turns_are_fair_between_owners_across_lanes(tmp_path):
    lanes = tmp_path / "queue"
    tickets = [
        # the render owner's batch joined first; the light owner has not had the GPU yet
        (
            lanes / "ram" / f"{1:020d}-1.json",
            {
                "who": "wide-w08-c001",
                "lane": "ram",
                "gpu": True,
                "state": "gpu-wait",
                "turn": "2026-10-21T10:00:00+00:00",
            },
        ),
        (
            lanes / "light" / f"{2:020d}-2.json",
            {
                "who": "w07-post-tts",
                "lane": "light",
                "gpu": True,
                "state": "gpu-wait",
                "turn": "2026-10-21T10:00:00+00:00",
            },
        ),
    ]
    render, tts = tickets[0][0], tickets[1][0]
    last = {"wide": int(1792560000 * 1e9)}  # the render owner had the GPU a moment ago
    assert [d["who"] for d in gpu_ahead(render, tickets, last)] == ["w07-post-tts"]
    assert gpu_ahead(tts, tickets, last) == []
    # a job that runs on the GPU, or already holds it, is ahead of everyone
    running = (
        lanes / "audio" / f"{3:020d}-3.json",
        {"who": "mmaudio", "gpu": True, "state": "running"},
    )
    assert [d["who"] for d in gpu_ahead(tts, tickets + [running], last)] == ["mmaudio"]


def test_cli_gpu_flag_and_environment(monkeypatch):
    assert _parser().parse_args(["run", "--gpu", "--", "true"]).gpu is True
    assert _parser().parse_args(["run", "--", "true"]).gpu is False
    monkeypatch.setenv("STAGELOAD_GPU", "1")
    assert _parser().parse_args(["run", "--", "true"]).gpu is True


# --- D and E: the owner of a run that just ended keeps the next turn --------------------


def _ticket(lane, ns, **data):
    return (lane / f"{ns:020d}-{ns}.json", {"state": "waiting", **data})


def test_the_owner_of_a_finished_run_takes_the_next_turn_with_a_command_sent_after_it(tmp_path):
    lane = tmp_path / "queue" / "ram"
    other = _ticket(lane, 100, who="w07-post-release")
    batch = _ticket(lane, 150, who="box-s1-checks")  # queued before the fit ended: no hold
    fit_ended = 200
    build = _ticket(lane, 300, who="box-s1-build")  # sent after the fit ended: the hold is its
    hold = {"owner": "box", "who": "box-s1-fit", "ended_ns": fit_ended, "until": time.time() + 20}
    live = [other, batch, build]
    assert lane_ahead(build[0], live, {}, hold) == ([], "")
    ahead, reason = lane_ahead(other[0], live, {}, hold)
    assert reason.startswith("held") and "box" in reason
    assert lane_ahead(batch[0], live, {}, hold)[1].startswith("held")
    # without the hold the older entry goes first, as before
    assert lane_ahead(other[0], live, {}, None) == ([], "")
    assert lane_ahead(build[0], live, {}, None)[1].startswith("2 ahead")


def test_a_deadline_of_another_owner_still_wins_over_the_hold(tmp_path):
    lane = tmp_path / "queue" / "ram"
    due = _ticket(lane, 100, who="w07-release", deadline="2026-10-21T20:00Z")
    build = _ticket(lane, 300, who="box-s1-build")
    hold = {"owner": "box", "who": "box-s1-fit", "ended_ns": 200, "until": time.time() + 20}
    assert lane_ahead(due[0], [due, build], {}, hold) == ([], "")
    assert lane_ahead(build[0], [due, build], {}, hold)[1] == "1 ahead: w07-release"


def test_one_hold_per_finished_run_and_a_batch_still_takes_turns(tmp_path, monkeypatch):
    # the step sent after the end uses the hold up; an entry the owner queued before gets none
    monkeypatch.setenv("STAGELOAD_HOLD_S", "20")
    io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [], log=lambda line: None)
    ok = [sys.executable, "-c", "pass"]
    config = GuardConfig(poll=0.05, grace=1.0)
    assert run_queued(ok, config, io=io, home=tmp_path, who="box-s1-fit", trace=None) == 0
    hold = json.loads((tmp_path / "queue" / "ram" / "hold.json").read_text())
    assert hold["owner"] == "box" and hold["until"] > time.time() + 15
    started = time.monotonic()
    assert run_queued(ok, config, io=io, home=tmp_path, who="box-s1-build", trace=None) == 0
    assert time.monotonic() - started < 5  # took the held turn at once
    hold = json.loads((tmp_path / "queue" / "ram" / "hold.json").read_text())
    assert hold["who"] == "box-s1-build"  # its own end set the next hold


def test_a_job_of_another_owner_waits_for_the_hold_then_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_HOLD_S", "1")
    logs = []
    io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [], log=logs.append)
    ok = [sys.executable, "-c", "pass"]
    config = GuardConfig(poll=0.05, grace=1.0)
    assert run_queued(ok, config, io=io, home=tmp_path, who="box-s1-fit", trace=None) == 0
    assert run_queued(ok, config, io=io, home=tmp_path, who="w07-post", trace=None) == 0
    assert any("held" in line and "box-s1-fit just ended" in line for line in logs)


def test_which_failures_hold_the_lane_longer():
    assert _quick_failure(1, 2.0)
    assert _quick_failure(127, 0.1)  # the command could not even start
    assert not _quick_failure(0, 2.0)
    assert not _quick_failure(1, 60.0)
    assert not _quick_failure(3, 2.0)  # killed by the guard
    assert not _quick_failure(143, 2.0)  # stopped by a signal


def test_hold_lengths(monkeypatch):
    # 29.10: no hold after a run that ended well, only after a quick failure
    monkeypatch.delenv("STAGELOAD_HOLD_S")
    assert _hold_seconds(False) == HOLD_S == 0.0
    assert _hold_seconds(True) == QUICK_FAIL_HOLD_S == 30.0
