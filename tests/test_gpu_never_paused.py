"""A --gpu command is never paused; a paused command resumes without waiting on its own swap; every
pause, kill and verdict names the largest footprints.

2026-10-24, 06:22-06:25Z: something outside the queue grew some 14 GB (compressor 6.6 -> 16.3 GB,
level 43 -> 23) while a GPU video render, the only ram job, stayed flat. The guard
paused it as outside pressure. Stopped, its Metal buffers were unwired (wired 20 -> 3 GB),
compressed and swapped: compressor 26 GB, swap 12.5 GB. The resume gate wanted swap under 6 GB,
which the paused job's own pages in swap could not allow, so a paused GPU job could only end in
the 10-minute kill.
"""

import sys
import time

import pytest

from stageload import queue
from stageload.guard import (
    EXIT_KILLED,
    Growth,
    GuardConfig,
    GuardIO,
    _process_name,
    queue_culprit,
    run_guarded,
)
from stageload.queue import run_queued

GB = 2**30
MB = 2**20
GATE = {"STAGELOAD_GATE_SWAP_GB": "6", "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35", "STAGELOAD_PAUSE_SWAP_GB": "9",
        "STAGELOAD_PAUSE_COMPRESSOR_GB": "12", "STAGELOAD_PAUSE_LEVEL": "25"}
WORK = [sys.executable, "-c", "import time\nfor _ in range(30): time.sleep(0.05)"]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
TOP = [(42755, 22 * GB, "zsh /Users/me/channel2-growth/tools/h3_exec_convrot.sh --profile"),
       (9001, 14 * GB, "nice -n 15 /Users/me/local-llm/mmaudio-venv/bin/python build_w09.py"),
       (851, int(0.4 * GB), "/Users/me/Applications/Claude.app/Contents/MacOS/Claude")]


@pytest.fixture
def gate(monkeypatch):
    for name, value in GATE.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


class Machine:
    """06:22Z readings once ``bad`` is set: the compressor 16.3 GB, the level 23 %, swap empty."""

    def __init__(self, level=23, swap=0, outside=30 * GB):
        self.started = None
        self.bad = dict(level=level, swap=swap, compressor=int(16.3 * GB))
        self.good = dict(level=70, swap=0, compressor=2 * GB)
        self.outside = outside
        self.logs = []

    def now(self, name):
        if self.started is None or time.monotonic() - self.started < 0.3:
            return self.good[name]
        return self.bad[name]

    def io(self):
        def swap():
            if self.started is None and any("start" in line for line in self.logs):
                self.started = time.monotonic()
            return self.now("swap")

        return GuardIO(level=lambda: self.now("level"), swap_used=swap, processes=lambda: [],
                       log=self.logs.append, compressor=lambda: self.now("compressor"),
                       footprint=lambda pid: 22 * GB, outside=lambda: self.outside,
                       top=lambda: TOP, neighbours=lambda: [])


def _lines(logs, text):
    return [line for line in logs if text in line]


def _quick(**fields):
    return GuardConfig(poll=0.05, grace=1.0, **fields)


# --- a --gpu command -----------------------------------------------------------------------


def test_a_gpu_job_runs_on_under_outside_pressure(tmp_path, gate):
    m = Machine()
    assert run_queued(WORK, _quick(pause_max=30), io=m.io(), home=tmp_path, gpu=True,
                      who="code-isekai-omb-c104bLv4", trace=None) == 0
    assert not _lines(m.logs, "pausing") and not _lines(m.logs, "killing"), m.logs
    (ran_on,) = _lines(m.logs, "never paused")
    assert "compressor 16.3 GB > 12.0 GB machine-wide" in ran_on
    assert "memory available 23% < 25% machine-wide" in ran_on
    assert "stops at once under 15% available" in ran_on
    assert ("pid 42755 zsh h3_exec_convrot.sh 22.0 GB; pid 9001 python build_w09.py 14.0 GB"
            in ran_on)


def test_a_gpu_job_under_the_kill_floor_is_killed_without_a_pause(tmp_path, gate):
    m = Machine(level=12)
    assert run_queued(SLEEP, _quick(pause_max=30), io=m.io(), home=tmp_path, gpu=True,
                      who="code-isekai-omb-c104bLv4", trace=None) == EXIT_KILLED
    assert not _lines(m.logs, "pausing")
    (kill,) = _lines(m.logs, "killing pid")
    assert "memory available 12% < 15% twice" in kill and "pid 9001 python build_w09.py" in kill


def test_a_gpu_job_over_its_swap_budget_runs_on_while_memory_is_available(tmp_path, gate):
    m = Machine(level=40, swap=int(12.5 * GB))
    assert run_queued(SLEEP, _quick(pause_max=30, swap_budget=8 * GB), io=m.io(),
                      home=tmp_path, gpu=True, who="box-c104", trace=None) == 0
    assert not _lines(m.logs, "pausing")
    assert not _lines(m.logs, "killing pid")
    assert len(_lines(m.logs, "a --gpu command runs on")) == 1


def test_a_gpu_job_over_its_swap_budget_and_the_ceiling_is_killed_without_a_pause(tmp_path, gate):
    m = Machine(level=40, swap=int(25 * GB))
    assert run_queued(SLEEP, _quick(pause_max=30, swap_budget=8 * GB), io=m.io(),
                      home=tmp_path, gpu=True, who="box-c104", trace=None) == EXIT_KILLED
    assert not _lines(m.logs, "pausing")
    assert "(swap 25.0 GB > 8.0 GB)" in _lines(m.logs, "killing pid")[0]


def test_a_gpu_job_over_its_swap_budget_under_the_swap_level_is_killed(tmp_path, gate):
    m = Machine(level=20, swap=int(12.5 * GB))
    assert run_queued(SLEEP, _quick(pause_max=30, swap_budget=8 * GB), io=m.io(),
                      home=tmp_path, gpu=True, who="box-c104", trace=None) == EXIT_KILLED
    assert "(swap 12.5 GB > 8.0 GB)" in _lines(m.logs, "killing pid")[0]


def test_a_gpu_job_is_never_the_one_the_blame_falls_on():
    h3 = Growth("code-isekai-omb-c104bLv4", grown=20 * GB, recent=GB, rise=2 * GB,
                started=200.0, gpu=True)
    build = Growth("w09-draft", grown=3 * GB, recent=GB, rise=GB, started=100.0)
    # the build rose too: it is paused, though the render started later
    verdict, target, _ = queue_culprit(0, [Growth("ours", 0, 0, me=True), h3, build], 512 * MB)
    assert verdict == "theirs" and target is build
    verdict, target, _ = queue_culprit(0, [Growth("w09-draft", 0, 0, me=True, rise=GB,
                                                  started=100.0), h3], 512 * MB)
    assert verdict == "ours"
    # only the render rose: nobody else is paused for it; its own kill floor is what holds it
    verdict, target, _ = queue_culprit(0, [Growth("ours", 0, 0, me=True), h3], 512 * MB)
    assert verdict == "theirs" and target is h3


# --- the resume of other commands ----------------------------------------------------------


def test_a_paused_job_resumes_with_its_own_pages_in_swap_once_the_level_is_back(gate):
    m = Machine()
    resumed = []

    def level():
        if _lines(m.logs, "pausing pid"):
            resumed.append(1)
            if len(resumed) > 4:
                return 40  # back over the gate, while swap holds 12.5 GB of our pages
        return m.now("level")

    m.bad["swap"] = 0
    io = m.io()
    io.level = level
    swap = io.swap_used
    io.swap_used = lambda: int(12.5 * GB) if _lines(m.logs, "pausing pid") else swap()
    assert run_guarded(WORK, _quick(pause_max=30), io) == 0
    (pause,) = _lines(m.logs, "pausing pid")
    assert "Largest footprints: pid 42755 zsh h3_exec_convrot.sh 22.0 GB" in pause
    assert "Resuming once memory available is 25% or more or the footprint outside the queue, "\
           "30.0 GB now, falls 1.0 GB" in pause
    (resume,) = _lines(m.logs, "resuming pid")
    assert "memory available 40% >= 25%" in resume  # 29.10: the pause floor, not the gate
    assert not _lines(m.logs, "killing")


def test_a_paused_job_resumes_once_the_outside_footprint_falls(gate):
    m = Machine()
    io = m.io()
    calls = []

    def falling():
        calls.append(1)
        return 30 * GB if len(calls) < 4 else int(28.5 * GB)

    def level():
        # 23 % until the resume; 30 % after it, under the gate but over the pause floor
        return 30 if _lines(m.logs, "resuming pid") else m.now("level")

    io.outside, io.level = falling, level
    assert run_guarded(WORK, _quick(pause_max=30), io) == 0
    (resume,) = _lines(m.logs, "resuming pid")
    assert "the footprint outside the queue fell to 28.5 GB from 30.0 GB" in resume
    assert "memory available" not in resume  # the level was still 23 %


def test_a_pause_that_does_not_ease_still_ends_in_the_kill_after_the_pause_max(gate):
    m = Machine()
    assert run_guarded(SLEEP, _quick(pause_max=0.5), m.io()) == EXIT_KILLED
    kill = _lines(m.logs, "killing pid")[0]
    assert "still memory available 23% < 25%" in kill
    assert "the footprint outside the queue 30.0 GB against 30.0 GB at the pause" in kill
    assert "Largest footprints: pid 42755" in kill


# --- the listing ---------------------------------------------------------------------------


def test_process_names_are_the_program_and_its_script():
    assert _process_name(TOP[0][2]) == "zsh h3_exec_convrot.sh"
    assert _process_name(TOP[1][2]) == "python build_w09.py"
    assert _process_name("/Users/me/local-llm/h3-forks/lachlanstuart/h3 --profile") == "h3"
    assert _process_name("") == "?"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS footprints")
def test_the_listing_reads_the_machine():
    from stageload.guard import outside_footprint, top_footprint

    rows = top_footprint()
    assert 0 < len(rows) <= 5 and rows == sorted(rows, key=lambda r: r[1], reverse=True)
    assert outside_footprint() > 0
