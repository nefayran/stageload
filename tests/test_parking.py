"""A run far back in its lane parks as a small shell loop; what the loop carries and leaves.

29.10: the resumed waiter lost the deadline the cleaner had put into its ticket and fell back
behind every entry with one, and a parked loop ended by a signal left no history line, so
`stageload requeue` could not find it.
"""

import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from stageload.guard import GuardConfig
from stageload.queue import _PARK_LOOP, NEAR_FAR, parked_ended, run_spec, write_spec

TRUE = [sys.executable, "-c", "pass"]


def test_the_loop_is_valid_shell_and_tells_far_from_failure():
    assert subprocess.run(["/bin/sh", "-n", "-c", _PARK_LOOP]).returncode == 0
    assert NEAR_FAR not in (0, 1)  # 0: near, 1: a Python failure; far has a code of its own
    assert f"-eq {NEAR_FAR}" in _PARK_LOOP and "trap" in _PARK_LOOP


def test_a_resumed_run_keeps_when_it_was_first_queued(tmp_path):
    since = (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat(timespec="seconds")
    spec = write_spec(TRUE, GuardConfig(poll=0.05, start_timeout=3600, wait_free=0), home=tmp_path,
                      who="w11-c009", deadline="2000-01-01T00:00Z", since=since)
    data = json.loads(spec.read_text())
    assert data["since"] == since and data["deadline"] == "2000-01-01T00:00Z"
    assert run_spec(spec) == 0
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["who"] == "w11-c009" and record["waited_s"] >= 0



def test_a_parked_loop_ended_by_a_signal_leaves_a_history_line(tmp_path):
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    ticket = lane / f"{1:020d}-4242.json"
    ticket.write_text(json.dumps({"pid": 4242, "who": "wide-w08-c010", "lane": "ram",
                                  "cmd": "zsh h3.sh", "state": "waiting", "argv": ["zsh", "h3.sh"],
                                  "cwd": "/tmp", "run_args": ["--who", "wide-w08-c010"],
                                  "since": "2026-10-29T10:00:00+00:00", "parked": "x"}))
    spec = write_spec(["zsh", "h3.sh"], GuardConfig(), home=tmp_path, who="wide-w08-c010",
                      cwd="/tmp", run_args=["--who", "wide-w08-c010"])
    parked_ended(ticket, spec, 143)
    assert not ticket.exists() and not spec.exists()
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert (record["who"], record["exit"], record["state"]) == ("wide-w08-c010", 143, "waiting")
    assert record["argv"] == ["zsh", "h3.sh"] and record["run_args"] == ["--who", "wide-w08-c010"]
    assert record["waited_s"] > 0


def test_the_spec_alone_is_enough_for_the_history_line(tmp_path):
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    spec = write_spec(["zsh", "h3.sh"], GuardConfig(), home=tmp_path, who="sh64-c8", cwd="/tmp",
                      run_args=["--who", "sh64-c8"], since="2026-10-29T10:00:00+00:00")
    parked_ended(lane / f"{2:020d}-4243.json", spec, 130)
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert (record["who"], record["lane"], record["exit"]) == ("sh64-c8", "ram", 130)
    assert record["cmd"] == "zsh h3.sh" and record["owner"] == "sh64"


def _fake_py(tmp_path, near_rc):
    log = tmp_path / "calls"
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\n"
                    f"echo \"$*\" >> {log}\n"
                    f"[ \"$3\" = --near ] && exit {near_rc}\n"
                    "exit 0\n")
    fake.chmod(0o755)
    return fake, log


def _sleeping(fifo, proc, seconds=10):
    import time
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline and not fifo.exists():
        time.sleep(0.05)
    time.sleep(0.3)   # its cat is on the FIFO now


def test_the_loop_itself_reports_a_signal(tmp_path):
    """2026-11-04: `ended` was called with no arguments, so inside it $2, $4 and $5 were empty
    (a shell function has positional parameters of its own); every parked run that a restart of
    the Mac ended wrote no history line and died with «exit: : numeric argument required»."""
    import signal
    fake, log = _fake_py(tmp_path, NEAR_FAR)
    fifo = tmp_path / "t.fifo"
    proc = subprocess.Popen(["/bin/sh", "-c", _PARK_LOOP, str(fake), "LANE", "TICKET", "NS", "SPEC", str(fifo)])
    _sleeping(fifo, proc)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 143
    assert "-m stageload.queue --ended TICKET SPEC 143" in log.read_text()
    assert not fifo.exists()


def test_a_parked_run_sleeps_until_it_is_woken(tmp_path):
    """2026-11-19: 661 parked clips each woke every 60 s for a Python that read the whole lane (load 283 on the Mac).
    A parked run now starts nothing until the dispatcher writes to its FIFO; then it asks --near once and runs."""
    import os, time
    from stageload.queue import wake_path
    fake, log = _fake_py(tmp_path, 0)
    ticket = tmp_path / "ram" / "0001-42.json"
    ticket.parent.mkdir()
    fifo = wake_path(ticket)
    fifo.parent.mkdir()
    proc = subprocess.Popen(["/bin/sh", "-c", _PARK_LOOP, str(fake), str(ticket.parent), str(ticket), "NS", "SPEC", str(fifo)])
    _sleeping(fifo, proc)
    time.sleep(1.5)
    assert not log.exists()                       # asleep: not one --near, nothing started
    from stageload.queue import wake
    assert wake(ticket)
    assert proc.wait(timeout=10) == 0
    calls = log.read_text().splitlines()
    assert calls[0].startswith("-m stageload.queue --near") and calls[1] == "-m stageload.queue SPEC"
    assert not fifo.exists()


def test_wake_without_a_sleeper_is_false(tmp_path):
    from stageload.queue import wake
    assert wake(tmp_path / "ram" / "0001-1.json") is False


def test_dispatch_wakes_only_the_runs_near_the_head(tmp_path, monkeypatch):
    """one dispatcher pass reads the lane once and writes only to the FIFOs of parked runs near the head"""
    import os
    from stageload import queue
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    me = os.getpid()
    paths = []
    for i in range(20):
        p = lane / f"{i + 1:020d}-{me}.json"
        p.write_text(json.dumps({"who": f"clip-{i}", "pid": me, "state": "waiting", "parked": "2026-11-19T12:00:00Z"}))
        paths.append(p)
    monkeypatch.setattr(queue, "_alive", lambda pid: True)
    woke = []
    monkeypatch.setattr(queue, "wake", lambda t: woke.append(t.name) or True)
    out = queue.dispatch(tmp_path)
    near = queue.lane_slots("ram") + queue.NEAR_EXTRA
    assert woke == [p.name for p in paths[: near + 1]]       # runs with at most slots + NEAR_EXTRA ahead
    assert out[0] == "ram:clip-0"


def test_dispatch_checks_only_the_head_of_a_long_lane(tmp_path, monkeypatch):
    """2026-11-19: the exact check of each of 645 parked clips took 16 s of CPU a pass; only the lane's first places
    are checked now"""
    import os
    from stageload import queue
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    for i in range(300):
        (lane / f"{i + 1:020d}-{os.getpid()}.json").write_text(
            json.dumps({"who": f"clip-{i}", "pid": os.getpid(), "state": "waiting", "parked": "x"}))
    monkeypatch.setattr(queue, "_alive", lambda pid: True)
    monkeypatch.setattr(queue, "wake", lambda t: True)
    calls = []
    real = queue.near_in
    monkeypatch.setattr(queue, "near_in", lambda *a: calls.append(1) or real(*a))
    queue.dispatch(tmp_path)
    assert len(calls) <= 3 * (queue.lane_slots("ram") + queue.NEAR_EXTRA) + 10
