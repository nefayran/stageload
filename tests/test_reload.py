"""`stageload reload` starts a waiting run again on the installed code, in its place in the lane: a process keeps the
code it started with, so runs that waited through an upgrade still follow the old rules."""

import json
import os
import subprocess
import sys
import time

import pytest

from stageload.reload import procargs, reload, waiting_runs

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="reads argv and environment through KERN_PROCARGS2")

ENV = {"STAGELOAD_GATE_SWAP_GB": "1000", "STAGELOAD_GATE_COMPRESSOR_GB": "1000", "STAGELOAD_GATE_LEVEL": "0",
       "STAGELOAD_GATE_DISK_GB": "0", "STAGELOAD_RESERVE_GB": "0", "STAGELOAD_REMOTE_LANES": "win"}


def _blocker(home):
    """A running job of this test process at the head of lane light, so the runs behind it wait."""
    lane = home / "queue" / "light"
    lane.mkdir(parents=True, exist_ok=True)
    path = lane / f"{1:020d}-{os.getpid()}.json"
    path.write_text(json.dumps({"pid": os.getpid(), "who": "blocker", "lane": "light", "state": "running",
                                "mem_peak": 0}))
    return path


def _detached(home, who, log):
    """A `stageload run` whose parent is launchd, as the studio scripts start them; returns its pid."""
    env = {**os.environ, **ENV, "STAGELOAD_HOME": str(home), "MARK": who}
    cmd = (f"{sys.executable} -m stageload run --lane light --who {who} --what test --mem-peak 0.1G "
           f"--wait-free 0 --no-trace -- {sys.executable} -c 'print(\"ran\")' >> {log} 2>&1 & echo $!")
    return int(subprocess.run(["sh", "-c", cmd], env=env, capture_output=True, text=True, check=True).stdout)


@pytest.fixture(autouse=True)
def _end_left_runs(tmp_path):
    yield
    for path in (tmp_path / "queue").glob("*/*.json"):
        try:
            pid = json.loads(path.read_text()).get("pid")
            if pid and pid != os.getpid():
                os.kill(pid, 15)
        except (OSError, ValueError):
            pass  # gone meanwhile


def _tickets(home):
    return sorted(p.name for p in (home / "queue" / "light").glob("*.json"))


def _wait(cond, timeout=20):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        if cond():
            return True
        time.sleep(0.1)
    return False


def test_procargs_reads_argv_and_environment():
    argv, env = procargs(os.getpid())
    assert argv[0].endswith(("python", "python3")) or "python" in argv[0]
    assert env.get("HOME") == os.environ.get("HOME")


def test_a_waiting_run_comes_back_in_its_place_with_its_environment_and_log(tmp_path):
    home = tmp_path
    blocker = _blocker(home)
    first, second = _detached(home, "t-first", tmp_path / "first.log"), None
    assert _wait(lambda: len(_tickets(home)) == 2)
    second = _detached(home, "t-second", tmp_path / "second.log")
    assert _wait(lambda: len(_tickets(home)) == 3)
    before = _tickets(home)
    rows = [r for r in waiting_runs(home) if r["who"] == "t-first"]
    done = reload(home, rows, log=lambda line: None)
    assert len(done) == 1 and done[0]["pid"] == first
    assert _wait(lambda: not os.path.exists(f"/proc/{first}") and subprocess.run(
        ["ps", "-p", str(first)], capture_output=True).returncode != 0)
    after = _tickets(home)
    new_pid = done[0]["new_pid"]
    # same time, new pid: still between the blocker and t-second
    assert after == [before[0], f"{before[1].split('-')[0]}-{new_pid}.json", before[2]]
    assert json.loads((home / "queue" / "light" / after[1]).read_text())["who"] == "t-first"
    assert procargs(new_pid)[1].get("MARK") == "t-first"
    cancelled = [json.loads(x) for x in (home / "cancelled.jsonl").read_text().splitlines()]
    assert cancelled[-1]["reason"] == "reload" and cancelled[-1]["pid"] == first
    blocker.unlink()
    assert _wait(lambda: "ran" in (tmp_path / "first.log").read_text() and "ran" in (tmp_path / "second.log").read_text(),
                 timeout=40)
    assert "stageload reload" in (tmp_path / "first.log").read_text()
    history = [json.loads(x) for x in (home / "history.jsonl").read_text().splitlines()]
    assert [(h["who"], h["exit"]) for h in history if h["who"] == "t-first"][-1] == ("t-first", 0)


def test_a_run_whose_parent_waits_for_it_is_left_alone(tmp_path):
    home = tmp_path
    _blocker(home)
    env = {**os.environ, **ENV, "STAGELOAD_HOME": str(home)}
    child = subprocess.Popen([sys.executable, "-m", "stageload", "run", "--lane", "light", "--who", "t-child",
                              "--what", "test", "--mem-peak", "0.1G", "--wait-free", "0", "--no-trace", "--", sys.executable, "-c", "pass"],
                             env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert _wait(lambda: len(_tickets(home)) == 2)
        lines = []
        assert reload(home, waiting_runs(home), log=lines.append) == []
        assert any(line.startswith("left") and "waits for its exit" in line for line in lines)
        assert child.poll() is None
    finally:
        child.terminate()
        child.wait(10)


def test_a_wrapper_round_its_run_is_started_again_whole(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_RELOAD_WRAPPERS", "render_shot.py:--rewrite")
    """A wrapper script waits for its run and handles its output: the wrapper is what starts again, and a later run of the same
    wrapper under another who (a later step) takes a time of its own."""
    home = tmp_path
    blocker = _blocker(home)
    wrapper = tmp_path / "render_shot.py"
    wrapper.write_text(
        "import os, subprocess, sys\n"
        "base = [sys.executable, '-m', 'stageload', 'run', '--lane', 'light', '--mem-peak', '0.1G', '--wait-free', '0',"
        " '--no-trace', '--what', 'test']\n"
        "subprocess.run(base + ['--who', 't-shot', '--', sys.executable, '-c', 'print(\"shot\")'], check=True)\n"
        "subprocess.run(base + ['--who', 't-shot-up', '--', sys.executable, '-c', 'print(\"up\")'], check=True)\n"
        "print('landed', flush=True)\n")
    env = {**os.environ, **ENV, "STAGELOAD_HOME": str(home), "MARK": "wrapped"}
    log = tmp_path / "wrapper.log"
    subprocess.run(["sh", "-c", f"{sys.executable} {wrapper} >> {log} 2>&1 &"], env=env, check=True, cwd=tmp_path)
    assert _wait(lambda: len(_tickets(home)) == 2)
    shot = [r for r in waiting_runs(home) if r["who"] == "t-shot"]
    ns = shot[0]["ns"]
    done = reload(home, shot, log=lambda line: None)
    assert len(done) == 1
    assert procargs(done[0]["new_pid"])[0][-1].endswith("render_shot.py")
    assert procargs(done[0]["new_pid"])[1].get("MARK") == "wrapped"
    assert any(name.startswith(ns) for name in _tickets(home))
    blocker.unlink()
    assert _wait(lambda: "landed" in log.read_text(), timeout=40)
    history = [json.loads(x) for x in (home / "history.jsonl").read_text().splitlines()]
    assert ("t-shot", 0) in [(h["who"], h["exit"]) for h in history]
    assert ("t-shot-up", 0) in [(h["who"], h["exit"]) for h in history]
    assert "failed" not in log.read_text()


def test_end_for_retry_keeps_the_place_for_the_scripts_own_retry(tmp_path):
    """Scripts that queue the same --who again when a run was ended before its start: --end-for-retry ends the run
    and the retry takes the old ticket's time."""
    home = tmp_path
    blocker = _blocker(home)
    env = {**os.environ, **ENV, "STAGELOAD_HOME": str(home)}
    run = (f"{sys.executable} -m stageload run --lane light --who t-retry --what test --mem-peak 0.1G --wait-free 0 "
           f"--no-trace -- {sys.executable} -c 'print(\"ran\")'")
    script = f"for i in 1 2 3; do {run} && break; sleep 0.5; done; echo end $?"
    log = tmp_path / "retry.log"
    parent = subprocess.Popen(["sh", "-c", script], env=env, stdout=log.open("w"), stderr=subprocess.STDOUT)
    try:
        assert _wait(lambda: len(_tickets(home)) == 2)
        later = _detached(home, "t-later", tmp_path / "later.log")
        assert _wait(lambda: len(_tickets(home)) == 3)
        before = _tickets(home)
        rows = [r for r in waiting_runs(home) if r["who"] == "t-retry"]
        assert reload(home, rows, log=lambda line: None) == []            # left alone without the flag
        done = reload(home, rows, end_for_retry=True, log=lambda line: None)
        assert done and done[0]["reason"] == "reload-for-retry"
        ns = before[1].split("-")[0]
        assert _wait(lambda: any(n.startswith(ns) and not n.endswith(f"-{rows[0]['pid']}.json") for n in _tickets(home)))
        after = _tickets(home)
        assert after[0] == before[0] and after[1].startswith(ns) and after[2] == before[2]
        assert not (home / "places").exists() or not list((home / "places").iterdir())
        blocker.unlink()
        assert _wait(lambda: "ran" in log.read_text() and "ran" in (tmp_path / "later.log").read_text(), timeout=40)
    finally:
        parent.terminate()
        parent.wait(10)
        del later
