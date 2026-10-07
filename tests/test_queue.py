import json
import os
import signal
import subprocess
import sys
import time

from stageload.cli import main
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import list_queue, run_queued

MB = 2**20

RUN = """
import sys
from pathlib import Path
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import run_queued
io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [],
             log=lambda line: print(line, file=sys.stderr, flush=True))
sys.exit(run_queued({cmd!r}, GuardConfig(poll=0.05, grace=1.0), io=io, home=Path({home!r}),
                    lane={lane!r}, who={who!r}, trace=None))
"""


def _job(log, who, seconds):
    """A command that writes when it starts and ends into ``log``."""
    code = (
        "import pathlib, sys, time; p = pathlib.Path(sys.argv[1]); "
        "f = p.open('a'); f.write(f'start {sys.argv[2]} {time.time()}\\n'); f.flush(); "
        f"time.sleep({seconds}); f.write(f'end {{sys.argv[2]}} {{time.time()}}\\n'); f.close()"
    )
    return [sys.executable, "-c", code, str(log), who]


def _start_run(home, log, who, seconds, lane="ram"):
    code = RUN.format(cmd=_job(log, who, seconds), home=str(home), lane=lane, who=who)
    return subprocess.Popen([sys.executable, "-c", code], stderr=subprocess.PIPE, text=True)


def _events(log):
    out = []
    for line in log.read_text().splitlines():
        kind, who, t = line.split()
        out.append((kind, who, float(t)))
    return out


def _wait_for(predicate, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError("condition never came true")


def _started(log, who):
    return lambda: log.exists() and f"start {who} " in log.read_text()


def _quiet_io(levels=(90,), processes=lambda: []):
    logs = []
    it = iter(levels)
    last = [levels[0]]

    def level():
        last[0] = next(it, last[0])
        return last[0]

    return GuardIO(level=level, swap_used=lambda: 0, processes=processes, log=logs.append), logs


def test_two_runs_in_one_lane_never_overlap(tmp_path):
    log = tmp_path / "log"
    first = _start_run(tmp_path, log, "A", 0.6)
    _wait_for(_started(log, "A"))
    second = _start_run(tmp_path, log, "B", 0.1)
    assert first.wait(timeout=20) == 0
    assert second.wait(timeout=20) == 0
    times = {(kind, who): t for kind, who, t in _events(log)}
    assert times[("end", "A")] <= times[("start", "B")]


def test_waiting_runs_start_in_the_order_they_joined(tmp_path):
    log = tmp_path / "log"
    runs = [_start_run(tmp_path, log, "A", 1.0)]
    _wait_for(_started(log, "A"))
    runs.append(_start_run(tmp_path, log, "B", 0.1))
    time.sleep(0.4)
    runs.append(_start_run(tmp_path, log, "C", 0.1))
    for run in runs:
        assert run.wait(timeout=30) == 0
    starts = [who for kind, who, _ in _events(log) if kind == "start"]
    assert starts == ["A", "B", "C"]


def test_a_ticket_left_by_a_dead_process_does_not_block_the_queue(tmp_path):
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    (lane / f"{1:020d}-{dead.pid}.json").write_text(json.dumps({"pid": dead.pid, "who": "gone"}))
    log = tmp_path / "log"
    run = _start_run(tmp_path, log, "A", 0.0)
    assert run.wait(timeout=10) == 0
    assert _started(log, "A")()


def test_lanes_do_not_wait_for_each_other(tmp_path):
    log = tmp_path / "log"
    heavy = _start_run(tmp_path, log, "A", 1.5, lane="ram")
    _wait_for(_started(log, "A"))
    remote = _start_run(tmp_path, log, "B", 0.0, lane="codex")
    assert remote.wait(timeout=10) == 0
    assert heavy.wait(timeout=20) == 0
    times = {(kind, who): t for kind, who, t in _events(log)}
    assert times[("start", "B")] < times[("end", "A")]


def test_the_queue_lists_the_running_job_and_the_waiting_ones(tmp_path):
    log = tmp_path / "log"
    first = _start_run(tmp_path, log, "sayu-ltx", 1.5)
    _wait_for(_started(log, "sayu-ltx"))
    second = _start_run(tmp_path, log, "haraiya-rife", 0.0)
    _wait_for(lambda: len(list_queue(tmp_path)) == 2)
    rows = list_queue(tmp_path)
    assert [(r["state"], r["who"], r["lane"]) for r in rows] == [
        ("running", "sayu-ltx", "ram"),
        ("waiting", "haraiya-rife", "ram"),
    ]
    assert first.wait(timeout=20) == 0
    assert second.wait(timeout=20) == 0
    assert list_queue(tmp_path) == []


def test_a_waiting_run_that_is_terminated_never_starts_its_command(tmp_path):
    log = tmp_path / "log"
    first = _start_run(tmp_path, log, "A", 2.0)
    _wait_for(_started(log, "A"))
    second = _start_run(tmp_path, log, "B", 0.0)
    _wait_for(lambda: len(list_queue(tmp_path)) == 2)
    second.send_signal(signal.SIGTERM)
    assert second.wait(timeout=5) == 128 + signal.SIGTERM
    assert [r["who"] for r in list_queue(tmp_path)] == ["A"]
    assert first.wait(timeout=20) == 0
    assert "start B" not in log.read_text()


def test_the_trace_counts_memory_of_the_command_children_too(tmp_path):
    holder = "import time; b = b'x' * (300 * 2**20); time.sleep(2.0)"
    parent = f"import subprocess, sys; subprocess.run([sys.executable, '-c', {holder!r}])"
    trace = tmp_path / "trace.jsonl"
    io, _ = _quiet_io()
    code = run_queued([sys.executable, "-c", parent], GuardConfig(poll=0.05, grace=1.0), io=io,
                      home=tmp_path, who="mem", trace=trace)
    assert code == 0
    records = [json.loads(line) for line in trace.read_text().splitlines()]
    peak = max(r["footprint"] for r in records if r["type"] == "sample")
    assert peak > 250 * MB


def test_every_finished_run_is_written_to_the_history(tmp_path):
    io, _ = _quiet_io()
    # long enough for the meter to sample it: a command gone before the first sample has no peak
    cmd = [sys.executable, "-c", "import sys, time; time.sleep(0.5); sys.exit(5)"]
    code = run_queued(cmd, GuardConfig(poll=0.05, grace=1.0), io=io, home=tmp_path, who="hist",
                      trace=tmp_path / "t.jsonl")
    assert code == 5
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert (record["who"], record["lane"], record["exit"]) == ("hist", "ram", 5)
    assert record["peak_footprint"] > 0


def test_the_shared_busy_list_holds_the_start(tmp_path):
    (tmp_path / "busy.txt").write_text("# heavy engines\nfake-heavy-engine\n")
    snapshots = iter([[(4242, 1, "python fake-heavy-engine --render")], []])
    last = [[]]

    def processes():
        last[0] = next(snapshots, last[0])
        return last[0]

    io, logs = _quiet_io(processes=processes)
    code = run_queued([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, grace=1.0), io=io,
                      home=tmp_path, trace=None)
    assert code == 0
    assert any("busy: python fake-heavy-engine" in line for line in logs)


def test_the_memory_check_still_runs_inside_the_queue(tmp_path):
    io, logs = _quiet_io(levels=(20, 25, 90))
    code = run_queued([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, grace=1.0), io=io,
                      home=tmp_path, trace=None)
    assert code == 0
    assert any("memory available 20% < 40%" in line for line in logs)


def test_cli_run_passes_the_exit_code_through(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    cmd = [sys.executable, "-c", "import sys; sys.exit(7)"]
    assert main(["run", "--wait-free", "0", "--who", "cli", "--", *cmd]) == 7
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["who"] == "cli"


def test_cli_queue_prints_an_empty_queue_as_json(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["queue", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_cli_run_needs_a_command(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    try:
        main(["run", "--"])
    except SystemExit as exit_:
        assert exit_.code == 2
    else:
        raise AssertionError("run without a command did not stop")
    assert "needs a command" in capsys.readouterr().err


def test_runs_from_a_shell_script_are_queued_too(tmp_path):
    # agents start jobs from shell scripts; the env var is all they need to share one home
    log = tmp_path / "log"
    env = {**os.environ, "STAGELOAD_HOME": str(tmp_path)}
    script = (f"{sys.executable} -m stageload run --wait-free 0 --who sh -- "
              f"{sys.executable} -c \"import pathlib; pathlib.Path('{log}').write_text('ok')\"")
    assert subprocess.run(["/bin/sh", "-c", script], env=env, timeout=30).returncode == 0
    assert log.read_text() == "ok"


def test_the_lane_stays_taken_while_the_command_of_a_killed_run_still_runs(tmp_path):
    # SIGKILL cannot be caught: the queued command goes on running, so it holds the lane itself
    log = tmp_path / "log"
    first = _start_run(tmp_path, log, "A", 1.5)
    _wait_for(_started(log, "A"))
    first.kill()
    first.wait(timeout=5)
    second = _start_run(tmp_path, log, "B", 0.0)
    assert second.wait(timeout=20) == 0
    _wait_for(lambda: "end A" in log.read_text())
    times = {(kind, who): t for kind, who, t in _events(log)}
    assert times[("end", "A")] <= times[("start", "B")]
