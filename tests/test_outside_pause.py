"""Swap that a process outside the queue adds pauses the command instead of killing it.

2026-10-23: twice a Next.js dev server of 3-4 GB, started outside the queue in the middle of a GPU video
render, took swap past the render's 7 GB budget. The render's footprint stayed flat; the guard
killed it both times (some 45 minutes of work) and the dev server ran on.
"""

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

import pytest

from stageload.cli import main
from stageload.guard import (
    EXIT_KILLED,
    GuardConfig,
    GuardIO,
    pause_max_default,
    run_guarded,
    top_resident,
)
from stageload.queue import list_queue, run_queued

GB = 2**30
QUICK = dict(poll=0.05, grace=2.0, swap_budget=3 * GB)  # limit 13 GB from 10 GB, resume at 12
NEXT = (4242, int(3.8 * GB), "node /Users/me/allkeep/node_modules/.bin/next dev -p 3100")
# runs for 1.5 s of its own time: it ends only if a pause is resumed
WORK = [sys.executable, "-c", "import time\nfor _ in range(30): time.sleep(0.05)"]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]
# the line `stageload queue` prints, as the command center at :4420 reads it
CENTER = re.compile(
    r"^(\S+)\s+(\d+)\s+(running|waiting|gpu-wait|paused)\s+(\S+)(?:\s+\[gpu\])?\s+pid (\d+)\s+"
    r"since (\S+)"
)


@pytest.fixture(autouse=True)
def _resume_level(monkeypatch):
    """A pause resumes once the level is back at the pause floor (29.10: 25 % on the machine, no
    longer the start gate's 35 %); here 35 %, since the conftest sets the floor to 0 with the rest
    of the gate."""
    monkeypatch.setenv("STAGELOAD_PAUSE_LEVEL", "35")


def _stat(pid):
    out = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True)
    return out.stdout.strip()


class Machine:
    """Fake swap and footprint readings in phases the test moves through; real clock and sleep.

    ``outside`` is how much swap the neighbour added; ``own`` how much the command grew."""

    def __init__(self, base=10 * GB, footprint=17 * GB):
        self.base, self.footprint = base, footprint
        self.outside = 0
        self.own = 0
        self.level = 60
        self.logs = []
        self.seen_states = []  # (state of the command's process, queue rows) read while paused
        self.home = None
        self.on_swap = None

    def swap(self):
        if self.on_swap is not None:
            self.on_swap(self)
        paused = self.paused_pid()
        if paused is not None:
            rows = list_queue(self.home) if self.home is not None else []
            self.seen_states.append((_stat(paused), [r["state"] for r in rows]))
        return self.base + self.outside + self.own

    def paused_pid(self):
        for line in reversed(self.logs):
            if "resuming pid" in line:
                return None
            if "pausing pid" in line:
                return int(line.split("pausing pid ")[1].split()[0])
        return None

    def io(self):
        return GuardIO(
            level=lambda: self.level,
            swap_used=self.swap,
            processes=lambda: [],
            log=self.logs.append,
            footprint=lambda pid: self.footprint + self.own,
            neighbours=lambda: [NEXT],
        )


def _ago(seconds):
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat(timespec="seconds")


def _after(seconds):
    """A phase change some time after the command started."""
    began = time.monotonic()
    return lambda: time.monotonic() - began >= seconds


def test_outside_swap_pauses_the_command_and_resumes_it_when_swap_falls():
    m = Machine()
    neighbour_starts, neighbour_stops = _after(0.4), _after(1.2)

    def phases(machine):
        present = neighbour_starts() and not neighbour_stops()
        machine.outside = 6 * GB if present else 0
        machine.level = 30 if present else 60  # the neighbour also takes the free memory

    m.on_swap = phases
    began = time.monotonic()
    assert run_guarded(WORK, GuardConfig(**QUICK, pause_max=30), m.io()) == 0
    assert time.monotonic() - began >= 1.5 + 0.5  # it lost the time it was paused, nothing more
    pause = next(line for line in m.logs if "pausing pid" in line)
    assert "swap 16.0 GB > 13.0 GB" in pause
    assert "footprint 17.0 GB changed by +0.0 GB" in pause
    assert "next dev" in pause and "pid 4242 3.8 GB" in pause  # names the neighbour
    # not on swap: the paused command's own pages go there while it waits
    assert any("resuming pid" in line and "memory available 60% >= 35%" in line
               for line in m.logs)
    assert not any("killing pid" in line for line in m.logs)
    assert m.seen_states and all(state.startswith("T") for state, _ in m.seen_states)
    assert any("guard: exit 0 (paused" in line for line in m.logs)


def test_outside_swap_that_stays_kills_the_command_after_the_pause_max():
    m = Machine()
    start = _after(0.3)
    def phases(machine):
        machine.outside, machine.level = (6 * GB, 30) if start() else (0, 60)

    m.on_swap = phases
    began = time.monotonic()
    assert run_guarded(SLEEP, GuardConfig(**QUICK, pause_max=0.6), m.io()) == EXIT_KILLED
    # stopped at once although it was stopped with SIGSTOP: it gets SIGCONT after SIGTERM
    assert time.monotonic() - began < 0.3 + 0.6 + QUICK["grace"]
    assert any("pausing pid" in line for line in m.logs)
    kill = next(line for line in m.logs if "killing pid" in line)
    assert "still memory available 30% < 35%" in kill and "(paused" in kill


def test_swap_the_command_added_itself_kills_it_as_before():
    m = Machine()
    grow = _after(0.3)
    m.on_swap = lambda machine: setattr(machine, "own", 6 * GB if grow() else 0)
    assert run_guarded(SLEEP, GuardConfig(**QUICK, pause_max=30), m.io()) == EXIT_KILLED
    assert not any("pausing" in line for line in m.logs)
    assert any("killing pid" in line and "(swap 16.0 GB > 13.0 GB)" in line for line in m.logs)


def test_a_command_that_grew_a_little_while_swap_grew_a_lot_is_still_paused():
    # footprints move by some hundred MB on their own (the video renders of 2026-10-23: 17.0-17.3 GB)
    m = Machine()
    start = _after(0.3)

    def phases(machine):
        machine.outside = 6 * GB if start() else 0
        machine.level = 30 if start() else 60
        machine.footprint = 17 * GB + (300 * 2**20 if start() else 0)

    m.on_swap = phases
    assert run_guarded(SLEEP, GuardConfig(**QUICK, pause_max=0.3), m.io()) == EXIT_KILLED
    assert any("pausing pid" in line for line in m.logs)


def test_without_a_footprint_reader_or_with_pausing_off_nothing_changes():
    for io_args, config in (({"footprint": None}, GuardConfig(**QUICK, pause_max=30)),
                            ({}, GuardConfig(**QUICK, pause_max=0))):
        m = Machine()
        start = _after(0.3)
        m.on_swap = lambda machine, start=start: setattr(
            machine, "outside", 6 * GB if start() else 0
        )
        io = m.io()
        for name, value in io_args.items():
            setattr(io, name, value)
        assert run_guarded(SLEEP, config, io) == EXIT_KILLED
        assert not any("pausing" in line for line in m.logs)


def test_the_pause_max_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("STAGELOAD_PAUSE_MAX_S", raising=False)
    assert pause_max_default() == 600
    assert GuardConfig().pause_max == 600
    monkeypatch.setenv("STAGELOAD_PAUSE_MAX_S", "90")
    assert GuardConfig().pause_max == 90
    monkeypatch.setenv("STAGELOAD_PAUSE_MAX_S", "nonsense")
    assert pause_max_default() == 600


def test_a_paused_run_shows_as_paused_in_the_queue_and_the_history(tmp_path):
    m = Machine()
    m.home = tmp_path
    neighbour_starts, neighbour_stops = _after(0.4), _after(1.2)

    def phases(machine):
        present = neighbour_starts() and not neighbour_stops()
        machine.outside = 6 * GB if present else 0
        machine.level = 30 if present else 60

    m.on_swap = phases
    code = run_queued(WORK, GuardConfig(**QUICK, pause_max=30), io=m.io(), home=tmp_path,
                      who="box-s1-c003", trace=None)  # a --gpu run is never paused
    assert code == 0
    assert m.seen_states and all(rows == ["paused"] for _, rows in m.seen_states)
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["exit"] == 0 and record["paused_s"] >= 0.4
    assert any("queue: exit 0 after" in line and "(paused" in line for line in m.logs)


def test_the_queue_line_of_a_paused_run(tmp_path, monkeypatch, capsys):
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    ticket = {"pid": os.getpid(), "who": "box-s1-c003", "owner": "box", "lane": "ram",
              "cmd": "python3 h3_shot.py", "state": "paused", "gpu": True, "gpu_turn": True,
              "started": "2026-10-23T03:00:00+00:00", "paused_since": _ago(95)}
    (lane / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps(ticket))
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["queue"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert line.startswith("ram     0 paused   box-s1-c003 [gpu]    pid ")
    assert re.search(r"since 2026-10-23T03:00:00\+00:00  paused 9[56] s  python3 h3_shot.py$", line)
    found = CENTER.match(line)
    assert found and found.group(3) == "paused" and found.group(4) == "box-s1-c003"


def test_a_paused_command_whose_run_died_is_resumed_by_the_queue(tmp_path):
    # a guard killed with SIGKILL cannot resume its command; it must not stay stopped for good
    proc = subprocess.Popen(SLEEP, start_new_session=True)
    try:
        os.killpg(proc.pid, signal.SIGSTOP)
        deadline = time.monotonic() + 5
        while not _stat(proc.pid).startswith("T"):
            assert time.monotonic() < deadline
            time.sleep(0.02)
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        lane = tmp_path / "queue" / "ram"
        lane.mkdir(parents=True)
        ticket = lane / f"{1:020d}-{dead.pid}.json"
        ticket.write_text(json.dumps({"pid": dead.pid, "who": "x", "state": "paused",
                                      "pgid": proc.pid}))
        assert list_queue(tmp_path) == []
        assert not ticket.exists()
        deadline = time.monotonic() + 5
        while _stat(proc.pid).startswith("T"):
            assert time.monotonic() < deadline, "the orphaned command stayed stopped"
            time.sleep(0.02)
    finally:
        proc.kill()
        proc.wait()


@pytest.mark.skipif(sys.platform != "darwin", reason="ps columns as on macOS")
def test_the_neighbours_leave_out_the_queued_runs_and_what_they_started():
    holder = subprocess.Popen(
        [sys.executable, "-c", "b = bytearray(64 * 2**20); b[::4096] = b'x' * len(b[::4096]); "
         "import time; time.sleep(30)"]
    )
    try:
        time.sleep(1.0)
        everyone = {pid for pid, _, _ in top_resident(set(), count=100000)}
        outside = {pid for pid, _, _ in top_resident({os.getpid()}, count=100000)}
        assert holder.pid in everyone
        assert holder.pid not in outside and os.getpid() not in outside
    finally:
        holder.kill()
        holder.wait()
