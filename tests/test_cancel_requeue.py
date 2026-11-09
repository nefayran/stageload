"""`stageload cancel` ends only the runs it names; `stageload requeue` sends again the runs a
signal ended.

2026-10-26, 15:02 UTC: one agent ran `pkill -TERM -f "stageload.queue .*"` to end its own jobs. It
ended the queue process of every agent, 70 waiting runs and a video post job 28 min into its run, and
nothing on disk let them be queued again.
"""

import json
import os
import subprocess
import sys
import time

from stageload import queue
from stageload.cli import main
from stageload.queue import cancel, list_queue, matching_tickets, requeue, requeue_plan

RUN = """
import sys
from pathlib import Path
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import run_queued
io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [],
             log=lambda line: print(line, file=sys.stderr, flush=True))
sys.exit(run_queued({cmd!r}, GuardConfig(poll=0.05, grace=1.0), io=io, home=Path({home!r}),
                    lane={lane!r}, who={who!r}, trace=None, run_args={run_args!r}))
"""
SLEEP = [sys.executable, "-c", "import time; time.sleep(30)"]


def _start_run(home, who, lane="ram", cmd=SLEEP):
    code = RUN.format(cmd=cmd, home=str(home), lane=lane, who=who,
                      run_args=["--who", who, "--lane", lane])
    return subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.DEVNULL)


def _wait_for(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never came true")


def _states(home):
    return {row["who"]: row["state"] for row in list_queue(home)}


def _history(home):
    path = home / "history.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _group_alive(pgid):
    try:
        os.killpg(pgid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def test_cancel_ends_only_the_named_waiting_run(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_WHO", "fde-cleanup")
    runs = [_start_run(tmp_path, "box-c001")]
    try:
        _wait_for(lambda: _states(tmp_path).get("box-c001") == "running")
        runs += [_start_run(tmp_path, "w10-post"), _start_run(tmp_path, "drama-c004")]
        _wait_for(lambda: len(list_queue(tmp_path)) == 3)
        rows = matching_tickets(tmp_path, who="w10-*")
        assert [r["who"] for r in rows] == ["w10-post"]
        ended = cancel(tmp_path, rows, grace=1.0, log=lambda line: None)
        assert [e["who"] for e in ended] == ["w10-post"]
        assert runs[1].wait(timeout=10) == 143
        assert _states(tmp_path) == {"box-c001": "running", "drama-c004": "waiting"}
        (line,) = [json.loads(x) for x in (tmp_path / "cancelled.jsonl").read_text().splitlines()]
        assert (line["who"], line["owner"], line["lane"], line["state"], line["pid"]) == (
            "w10-post", "w10", "ram", "waiting", runs[1].pid)
        assert line["asker"] and line["asker_who"] == "fde-cleanup"
        # the waiting run leaves a history line, which the turns do not count as a start
        (record,) = [r for r in _history(tmp_path) if r["who"] == "w10-post"]
        assert (record["exit"], record["state"], record["run_s"]) == (143, "waiting", 0.0)
        assert record["run_args"] == ["--who", "w10-post", "--lane", "ram"]
        assert "w10" not in queue._history_starts(tmp_path / "queue" / "ram")
    finally:
        for run in runs:
            run.kill()
            run.wait()


def test_cancel_stops_the_whole_group_of_a_running_command(tmp_path):
    run = _start_run(tmp_path, "box-c002")
    try:
        _wait_for(lambda: any(r.get("pgid") for r in matching_tickets(tmp_path, owner="box")))
        (row,) = matching_tickets(tmp_path, owner="box")
        pgid = int(row["pgid"])
        assert _group_alive(pgid)
        assert [e["state"] for e in cancel(tmp_path, [row], grace=1.0, log=lambda x: None)] == [
            "running"]
        assert run.wait(timeout=10) == 143
        _wait_for(lambda: not _group_alive(pgid), timeout=5)
        assert [r["exit"] for r in _history(tmp_path)] == [143]
    finally:
        run.kill()
        run.wait()


def test_cancel_refuses_when_nothing_matches_and_dry_only_lists(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["cancel", "--who", "nobody-*"]) == 1
    assert "no queued run matches --who 'nobody-*'" in capsys.readouterr().err
    run = _start_run(tmp_path, "sh60-c003", lane="light")
    try:
        _wait_for(lambda: _states(tmp_path).get("sh60-c003") == "running")
        assert main(["cancel", "--owner", "sh60", "--lane", "ram"]) == 1  # wrong lane
        assert main(["cancel", "--owner", "sh60", "--dry"]) == 0
        assert "would end light  running  sh60-c003" in capsys.readouterr().out
        assert run.poll() is None and not (tmp_path / "cancelled.jsonl").exists()
        assert main(["cancel", "--who", "sh60-*", "--lane", "light"]) == 0
        assert "ended  light  running  sh60-c003" in capsys.readouterr().out
        assert run.wait(timeout=10) == 143
    finally:
        run.kill()
        run.wait()


def test_the_ticket_and_the_history_keep_how_to_run_it_again(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    for name in ("STAGELOAD_WHO", "STAGELOAD_OWNER", "STAGELOAD_DEADLINE", "STAGELOAD_GPU"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("STAGELOAD_WHAT", "Checking the requeue")
    monkeypatch.chdir(tmp_path)
    cmd = [sys.executable, "-c", "pass"]
    assert main(["run", "--who", "kz-check", "--lane", "light", "--wait-free", "0",
                 "--mem-peak", "1G", "--no-trace", "--", *cmd]) == 0
    (record,) = _history(tmp_path)
    assert record["argv"] == cmd and record["cwd"] == str(tmp_path)
    assert record["run_args"] == ["--who", "kz-check", "--lane", "light", "--wait-free", "0",
                                  "--mem-peak", "1G", "--no-trace",
                                  "--what", "Checking the requeue"]
    spec = queue.write_spec(cmd, queue.GuardConfig(), home=tmp_path, cwd="/x", run_args=["-a"])
    assert {k: json.loads(spec.read_text())[k] for k in ("cwd", "run_args")} == {
        "cwd": "/x", "run_args": ["-a"]}


def _record(who, exit_, end, **kw):
    return {"end": end, "lane": "ram", "who": who, "cmd": "sleep", "exit": exit_, "pid": 4242,
            "argv": ["true"], "cwd": "/", "run_args": ["--who", who], **kw}


def _write_history(home, records):
    (home / "history.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))


def test_requeue_picks_the_runs_a_signal_ended_and_nothing_else(tmp_path):
    since = queue._since("2026-10-26T15:00Z")
    old = _record("w10-old", 143, "2026-10-26T15:02:10+00:00")
    del old["run_args"]
    _write_history(tmp_path, [
        _record("w10-post", 143, "2026-10-26T15:02:05+00:00"),
        _record("box-c004", -15, "2026-10-26T15:02:06+00:00", state="waiting"),
        _record("box-c005", 137, "2026-10-26T14:59:00+00:00"),  # before --since
        _record("box-c006", 143, "2026-10-26T15:02:07+00:00"),
        _record("box-c006", 0, "2026-10-26T15:30:00+00:00"),  # ran again and finished
        _record("box-c007", 1, "2026-10-26T15:02:08+00:00"),  # failed on its own
        _record("drama-c001", 130, "2026-10-26T15:02:09+00:00"),
        _record("sh61-c001", 143, "2026-10-26T15:02:09+00:00", pid=777),  # cancelled
        old,
    ])
    (tmp_path / "cancelled.jsonl").write_text(json.dumps({"who": "sh61-c001", "pid": 777}) + "\n")
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    (lane / f"{time.time_ns():020d}-{os.getpid()}.json").write_text(json.dumps(
        {"pid": os.getpid(), "who": "drama-c001", "state": "waiting"}))  # already queued again
    plan = requeue_plan(tmp_path, since)
    assert [(r["who"], why) for r, why in plan] == [
        ("w10-post", ""), ("box-c004", ""),
        ("w10-old", "no run_args (older than this version)")]
    assert [r["who"] for r, _ in requeue_plan(tmp_path, since, owner="box")] == ["box-c004"]
    lines = []
    assert requeue(tmp_path, plan, dry=True, log=lines.append) == []
    assert lines[0].startswith("would requeue  ram    exit 143  w10-post")
    assert lines[2].startswith("cannot requeue: no run_args (older than this version)")
    assert not (tmp_path / "requeued").exists()


def test_requeue_starts_the_run_again_detached_in_its_cwd(tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.setenv("STAGELOAD_WHO", "someone-else")  # the requeuer's own: it must not leak in
    _write_history(tmp_path, [{
        **_record("w10-post", 143, "2026-10-26T15:02:05+00:00"),
        "cwd": str(work),
        "argv": [sys.executable, "-c", "import os, pathlib; pathlib.Path('out').write_text("
                 "os.environ['STAGELOAD_LANE'])"],
        "run_args": ["--who", "w10-post", "--lane", "light", "--wait-free", "0", "--no-trace",
                     "--mem-peak", "1G"]}])
    plan = requeue_plan(tmp_path, queue._since("2026-10-26T15:00:00"))
    lines = []
    (proc,) = requeue(tmp_path, plan, log=lines.append)
    assert proc.wait(timeout=30) == 0
    assert (work / "out").read_text() == "light"
    again = _history(tmp_path)[-1]
    assert (again["who"], again["exit"], again["cwd"]) == ("w10-post", 0, str(work))
    (log,) = (tmp_path / "requeued").iterdir()
    assert log.name.startswith("w10-post-") and str(log) in lines[0]
    assert requeue_plan(tmp_path, queue._since("2026-10-26T15:00:00")) == []  # it finished


def test_cli_requeue_dry_lists_the_old_records(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    old = _record("w10-post", 143, "2026-10-26T15:02:05+00:00")
    del old["run_args"]
    _write_history(tmp_path, [old])
    assert main(["requeue", "--since", "2026-10-26T15:00Z", "--dry"]) == 0
    assert "cannot requeue: no run_args (older than this version)" in capsys.readouterr().out
    assert main(["requeue", "--since", "2026-10-27T00:00Z"]) == 0
    assert "no run ended by a signal" in capsys.readouterr().out
    assert main(["requeue", "--since", "yesterday"]) == 2


def test_requeue_finds_waiting_runs_that_died_without_a_history_line(tmp_path):
    """2026-11-04: a restart of the Mac ended about 85 waiting runs before they could write their
    history line; queued.jsonl, written when a run joins the queue, lets requeue find them."""
    since = queue._since("2026-11-04T13:14Z")
    _write_history(tmp_path, [_record("sh73-c2", 0, "2026-11-04T13:10:00+00:00", pid=11)])
    rows = [
        {**_record("drama-y09a", None, "", pid=21), "queued": "2026-11-04T12:00:00+00:00"},  # lost
        {**_record("sh73-c2", None, "", pid=11), "queued": "2026-11-04T12:00:00+00:00"},  # it finished
        {**_record("old-c1", None, "", pid=31), "queued": "2026-10-30T12:00:00+00:00"},  # too old
    ]
    for r in rows:
        del r["end"], r["exit"]
    (tmp_path / "queued.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan = requeue_plan(tmp_path, since)
    assert [(r["who"], r["exit"], why) for r, why in plan] == [("drama-y09a", "lost", "")]


def test_a_run_writes_its_queued_line(tmp_path):
    from stageload.guard import GuardConfig
    from stageload.queue import run_spec, write_spec
    spec = write_spec([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, wait_free=0), home=tmp_path,
                      who="sh80-c1", run_args=["--who", "sh80-c1"], cwd=str(tmp_path))
    assert run_spec(spec) == 0
    (line,) = (tmp_path / "queued.jsonl").read_text().splitlines()
    assert json.loads(line)["who"] == "sh80-c1"
    assert requeue_plan(tmp_path, queue._since("2000-01-01T00:00Z")) == []  # it finished
