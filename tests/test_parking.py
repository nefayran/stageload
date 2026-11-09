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


def test_the_loop_itself_reports_a_signal(tmp_path):
    """2026-11-04: `ended` was called with no arguments, so inside it $2, $4 and $5 were empty
    (a shell function has positional parameters of its own); every parked run that a restart of
    the Mac ended wrote no history line and died with «exit: : numeric argument required»."""
    import os, signal, time
    log = tmp_path / "calls"
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\n"
                    f"echo \"$*\" >> {log}\n"
                    f"[ \"$3\" = --near ] && exit {NEAR_FAR}\n"
                    "exit 0\n")
    fake.chmod(0o755)
    loop = _PARK_LOOP.replace("sleep 60", "sleep 0.2")
    proc = subprocess.Popen(["/bin/sh", "-c", loop, str(fake), "LANE", "TICKET", "NS", "SPEC"])
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline and not (log.exists() and "--near" in log.read_text()):
        time.sleep(0.05)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=10) == 143
    assert "-m stageload.queue --ended TICKET SPEC 143" in log.read_text()
