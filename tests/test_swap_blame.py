"""Swap kills hit the job that added the memory, not a bystander; stopping a group never crashes
the guard.

2026-10-21: light jobs of 3-4 GB were killed twelve times for swap a 29 GB render beside them
added, and two
video renders lost their result when ``killpg`` raised EPERM inside the guard.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

import pytest

from stageload.guard import (
    EXIT_KILLED,
    Growth,
    GuardConfig,
    GuardIO,
    _stop_group,
    blame,
    critical_rank,
    run_guarded,
)
from stageload.queue import run_queued, running_jobs, trace_growth

GB = 2**30
MB = 2**20
macos = pytest.mark.skipif(
    sys.platform != "darwin", reason="reads macOS footprints and killpg semantics"
)
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
SHORT = [sys.executable, "-c", "import time; time.sleep(1.0)"]


def _io(swaps, levels=(90,), jobs=None):
    """Fake readings that move on by one value per call and then stay at the last one."""
    logs = []

    def reader(values):
        it, last = iter(values), [values[0]]

        def read():
            last[0] = next(it, last[0])
            return last[0]

        return read

    return GuardIO(
        level=reader(levels),
        swap_used=reader(swaps),
        processes=lambda: [],
        log=logs.append,
        jobs=jobs,
    ), logs


RENDER = Growth("render-c009", grown=28 * GB, recent=0)
LIGHT = Growth("w07-post-release", grown=4 * GB, recent=3 * GB, me=True)
QUICK = dict(poll=0.05, grace=1.0, swap_budget=3 * GB)


# --- A: the decision itself ---------------------------------------------------------------


def test_a_small_job_beside_a_big_one_owns_only_its_share_of_the_swap():
    share, cause = blame([RENDER, LIGHT])
    assert share == pytest.approx(4 / 32)
    assert cause is RENDER


def test_a_job_alone_or_unseen_owns_all_the_swap_as_before():
    assert blame([LIGHT]) == (1.0, None)
    assert blame([RENDER]) == (1.0, None)  # our own trace is unreadable: act as before
    assert blame([]) == (1.0, None)


def test_the_job_that_added_most_is_the_cause():
    me = Growth("h3", grown=28 * GB, recent=1 * GB, me=True)
    share, cause = blame([me, Growth("light", grown=4 * GB, recent=3 * GB)])
    assert cause is None
    assert share == pytest.approx(28 / 32)


def test_the_biggest_recent_grower_goes_first_when_memory_runs_out():
    assert (
        critical_rank([RENDER, LIGHT]) == 0
    )  # the light job grew 3 GB in the last minute, the render none
    tts = Growth("tts", grown=12 * GB, recent=12 * GB)
    assert critical_rank([RENDER, LIGHT, tts]) == 1


# --- A: the guard acting on it ----------------------------------------------------------


def test_swap_another_job_added_does_not_kill_ours():
    # 13.3 GB at our start, 16.3 GB a minute later: over our 3 GB budget, but the render explains it
    io, logs = _io(swaps=(13 * GB, 13 * GB, int(16.3 * GB)), jobs=lambda: [RENDER, LIGHT])
    assert run_guarded(SHORT, GuardConfig(**QUICK), io) == 0
    assert any("not ours to stop" in line and "render-c009" in line for line in logs)


def test_a_job_alone_is_killed_for_swap_as_before():
    io, logs = _io(swaps=(13 * GB, 13 * GB, 17 * GB), jobs=lambda: [LIGHT])
    assert run_guarded(SLEEP, GuardConfig(**QUICK), io) == EXIT_KILLED
    assert any("swap 17.0 GB > 16.0 GB" in line for line in logs)


def test_the_job_that_added_the_memory_is_killed_for_swap():
    me = Growth("h3", grown=28 * GB, recent=1 * GB, me=True)
    io, logs = _io(
        swaps=(10 * GB, 10 * GB, 30 * GB),
        jobs=lambda: [me, Growth("light", grown=4 * GB, recent=3 * GB)],
    )
    assert run_guarded(SLEEP, GuardConfig(**QUICK), io) == EXIT_KILLED
    assert any("killing pid" in line for line in logs)


def test_a_spared_job_is_not_killed_once_the_cause_is_gone():
    # swap stays up after the render ends (macOS gives swap back slowly): the growth was not ours
    calls = {"n": 0}

    def jobs():
        calls["n"] += 1
        return [RENDER, LIGHT] if calls["n"] == 1 else [LIGHT]

    io, logs = _io(swaps=(13 * GB, 13 * GB, 17 * GB, 17 * GB, 17 * GB), jobs=jobs)
    assert run_guarded(SHORT, GuardConfig(**QUICK), io) == 0


def test_the_absolute_ceiling_still_stops_the_cause():
    # two jobs of about one size: the growth raises our limit past the ceiling, which still holds
    me = Growth("h3", grown=15 * GB, recent=1 * GB, me=True)
    io, _ = _io(swaps=(5 * GB, 5 * GB, 21 * GB), jobs=lambda: [me, Growth("x", 14 * GB, 0)])
    config = GuardConfig(poll=0.05, grace=1.0, swap_budget=14 * GB, swap_limit=20 * GB)
    assert run_guarded(SLEEP, config, io) == EXIT_KILLED


def test_when_memory_runs_out_the_recent_grower_stops_first_and_the_others_later():
    # we are second in line: we wait two samples more than the first
    tts = Growth("tts", grown=12 * GB, recent=12 * GB)
    io, logs = _io(swaps=(0,), levels=(90, 5), jobs=lambda: [tts, LIGHT])
    assert run_guarded(SLEEP, GuardConfig(poll=0.05, grace=1.0), io) == EXIT_KILLED
    assert any("1 job(s) that grew more recently stop first" in line for line in logs)
    assert any("5% < 10% 4 times" in line for line in logs)


def test_when_memory_runs_out_the_recent_grower_is_stopped_at_the_second_sample():
    io, logs = _io(swaps=(0,), levels=(90, 5), jobs=lambda: [RENDER, LIGHT])
    assert run_guarded(SLEEP, GuardConfig(poll=0.05, grace=1.0), io) == EXIT_KILLED
    assert any("5% < 10% twice" in line for line in logs)


def test_a_failing_reader_of_the_other_jobs_means_acting_alone():
    def broken():
        raise OSError("disk gone")

    io, logs = _io(swaps=(13 * GB, 13 * GB, 17 * GB), jobs=broken)
    assert run_guarded(SLEEP, GuardConfig(**QUICK), io) == EXIT_KILLED
    assert any("cannot read the other jobs" in line for line in logs)


# --- A: reading the other jobs from their traces ------------------------------------------


def _trace(path, footprints, *, age=0.0, interval=0.5):
    """A trace like MemoryMeter writes, whose last sample is ``age`` seconds old."""
    began = time.time() - age - interval * (len(footprints) - 1)
    utc = datetime.fromtimestamp(began, timezone.utc).isoformat(timespec="seconds")
    lines = [{"type": "start", "utc": utc, "interval": interval}]
    lines += [
        {"type": "sample", "t": i * interval, "footprint": fp, "swap_used": 0, "level": 50}
        for i, fp in enumerate(footprints)
    ]
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_trace_growth_reads_growth_since_start_and_over_the_last_minute(tmp_path):
    # 4 minutes at 1 GB, then the job grows to 9 GB over its last 30 s
    footprints = [GB] * 480 + [GB + i * (8 * GB // 60) for i in range(61)]
    grown, recent = trace_growth(_trace(tmp_path / "t.jsonl", footprints))
    assert grown == pytest.approx(8 * GB, rel=0.01)
    assert recent == pytest.approx(8 * GB, rel=0.01)


def test_a_job_being_stopped_keeps_its_blame_while_it_frees_memory(tmp_path):
    footprints = [0] + [28 * GB] * 100 + [0] * 10  # the render's footprint reads 0 once it exits
    grown, _ = trace_growth(_trace(tmp_path / "t.jsonl", footprints))
    assert grown == 28 * GB


def test_a_stale_or_broken_trace_says_nothing(tmp_path):
    assert trace_growth(_trace(tmp_path / "old.jsonl", [0, GB], age=120)) is None
    (tmp_path / "bad.jsonl").write_text("not json\n")
    assert trace_growth(tmp_path / "bad.jsonl") is None
    assert trace_growth(tmp_path / "missing.jsonl") is None


def test_running_jobs_reads_every_lane_and_finds_old_tickets_by_pid(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    me_pid, other_pid = os.getpid(), os.getppid()
    mine = _trace(traces / "20261021T120000Z-w07-post-release-x.jsonl", [0, 3 * GB])
    # a ticket of the stageload running today names no trace; its default trace ends with its pid
    _trace(traces / f"20261021T110000Z-wide-w09-c009-{other_pid}.jsonl", [0, 28 * GB])
    for lane, pid, data in (("light", me_pid, {"trace": str(mine)}), ("ram", other_pid, {})):
        lane_dir = tmp_path / "queue" / lane
        lane_dir.mkdir(parents=True)
        ticket = lane_dir / f"{1:020d}-{pid}.json"
        ticket.write_text(json.dumps({"pid": pid, "who": lane, "state": "running", **data}))
    me_ticket = tmp_path / "queue" / "light" / f"{1:020d}-{me_pid}.json"
    jobs = running_jobs(tmp_path, me_ticket)()
    assert {(j.who, j.grown, j.me) for j in jobs} == {
        ("light", 3 * GB, True),
        ("ram", 28 * GB, False),
    }
    assert blame(jobs)[1].who == "ram"


RUN = """
import pathlib, sys
from stageload.guard import GuardConfig, GuardIO, list_processes
from stageload.queue import run_queued
flag = pathlib.Path({flag!r})
io = GuardIO(level=lambda: 90, swap_used=lambda: 10 * 2**30 if flag.exists() else 0,
             processes=lambda: [],
             log=lambda line: print(line, file=sys.stderr, flush=True))
sys.exit(run_queued({cmd!r}, GuardConfig(poll=0.1, grace=1.0, swap_budget=2**30), io=io,
                    home=pathlib.Path({home!r}), lane={lane!r}, who={who!r},
                    trace=pathlib.Path({trace!r})))
"""


def _holder(marker, mb, seconds):
    return [
        sys.executable,
        "-c",
        f"import pathlib, time; b = bytearray({mb} * 2**20); "
        f"pathlib.Path({str(marker)!r}).write_text('1'); "
        f"time.sleep({seconds})",
    ]


@macos
def test_two_guards_in_two_lanes_kill_only_the_job_that_grew(tmp_path):
    # end to end, as on the machine: each guard in its own process reads the other's trace
    flag = tmp_path / "swap-grew"
    runs = {}
    for who, lane, mb, seconds in (("light-job", "light", 30, 4.0), ("ram-render", "ram", 500, 60)):
        marker = tmp_path / f"{who}.up"
        code = RUN.format(
            flag=str(flag),
            cmd=_holder(marker, mb, seconds),
            home=str(tmp_path),
            lane=lane,
            who=who,
            trace=str(tmp_path / "traces" / f"{who}.jsonl"),
        )
        runs[who] = (
            subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE, text=True),
            marker,
        )
    deadline = time.monotonic() + 30
    while not all(marker.exists() for _, marker in runs.values()):
        assert time.monotonic() < deadline
        time.sleep(0.05)
    time.sleep(1.5)  # both meters have sampled the memory by now
    flag.write_text("1")  # swap jumps 10 GB: over both budgets
    light, render = runs["light-job"][0], runs["ram-render"][0]
    assert render.wait(timeout=30) == EXIT_KILLED
    assert light.wait(timeout=30) == 0
    assert "not ours to stop" in light.stderr.read()
    assert "Traceback" not in render.stderr.read()


# --- C: stopping a group never crashes the guard ------------------------------------------


def _state(pid):
    out = subprocess.run(
        ["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True
    ).stdout
    return out.strip()


@macos
def test_killpg_on_a_group_of_exited_members_raises_eperm_on_macos():
    # the cause measured on 2026-10-21: a group whose members have all exited but are not reaped
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    deadline = time.monotonic() + 10
    while not _state(proc.pid).startswith("Z"):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    with pytest.raises(PermissionError):
        os.killpg(proc.pid, signal.SIGTERM)
    proc.wait()


def test_stopping_a_group_whose_leader_has_exited_does_not_raise():
    proc = subprocess.Popen([sys.executable, "-c", "pass"], start_new_session=True)
    deadline = time.monotonic() + 10
    while not _state(proc.pid).startswith("Z"):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    _stop_group(proc, grace=0.5)  # the old guard raised PermissionError here
    assert proc.returncode is not None


def test_when_killpg_refuses_every_live_member_is_signalled_on_its_own(tmp_path, monkeypatch):
    pidfile = tmp_path / "member.pid"
    member = (
        "import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)"
    )
    leader = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {member!r}]); time.sleep(60)"
    )
    proc = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
    deadline = time.monotonic() + 15
    while not (pidfile.exists() and pidfile.read_text()):
        assert time.monotonic() < deadline
        time.sleep(0.02)
    member_pid = int(pidfile.read_text())

    def refuse(pgid, signum):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "killpg", refuse)
    _stop_group(proc, grace=0.5)
    assert proc.returncode is not None
    deadline = time.monotonic() + 5
    while _state(member_pid) and not _state(member_pid).startswith("Z"):
        assert time.monotonic() < deadline, "the member that ignores SIGTERM survived"
        time.sleep(0.05)


def test_a_run_killed_for_swap_ends_cleanly_when_killpg_refuses(tmp_path, monkeypatch):
    # 2026-10-21: the guard died with a traceback and the run left no result
    monkeypatch.setattr(
        os, "killpg", lambda pgid, signum: (_ for _ in ()).throw(PermissionError(1, "EPERM"))
    )
    io, logs = _io(swaps=(0, 0, 0, 20 * GB))
    code = run_queued(
        SLEEP,
        GuardConfig(poll=0.05, grace=0.5, swap_budget=6 * GB),
        io=io,
        home=tmp_path,
        who="wide-w08-c041",
        trace=None,
    )
    assert code == EXIT_KILLED
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert (record["who"], record["exit"]) == ("wide-w08-c041", EXIT_KILLED)
