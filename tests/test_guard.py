import os
import signal
import subprocess
import sys
import time

from stageload.guard import (
    EXIT_CANNOT_START,
    EXIT_KILLED,
    EXIT_NO_ROOM,
    GuardConfig,
    GuardIO,
    _stop_group,
    ancestors,
    run_guarded,
)

GB = 2**30


def series(*values):
    it = iter(values)
    last = [values[0]]

    def read():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return read


def make_io(levels=(90,), swaps=(0,), processes=lambda: []):
    clock = [0.0]
    logs = []

    def sleep(seconds):
        clock[0] += seconds

    io = GuardIO(
        level=series(*levels),
        swap_used=series(*swaps),
        processes=processes,
        sleep=sleep,
        clock=lambda: clock[0],
        log=logs.append,
    )
    return io, logs


QUICK = dict(poll=0.05, grace=1.0)
PASS = [sys.executable, "-c", "pass"]
SLEEP = [sys.executable, "-c", "import time; time.sleep(60)"]


def test_waits_for_memory_then_runs_the_command():
    io, logs = make_io(levels=(30, 35, 90))
    assert run_guarded(PASS, GuardConfig(**QUICK), io) == 0
    assert any("memory available 30% < 40%" in line for line in logs)


def test_gives_up_when_there_is_no_room_in_time():
    io, logs = make_io(levels=(10,))
    code = run_guarded(PASS, GuardConfig(start_timeout=10, **QUICK), io)
    assert code == EXIT_NO_ROOM
    assert any("no room" in line for line in logs)


def test_waits_for_swap_headroom_under_the_limit():
    io, logs = make_io(swaps=(15 * GB, 15 * GB, 5 * GB))
    config = GuardConfig(swap_budget=8 * GB, swap_limit=20 * GB, **QUICK)
    assert run_guarded(PASS, config, io) == 0
    assert any("limit" in line for line in logs)


def test_waits_while_a_busy_process_runs_but_not_for_itself():
    own = (os.getpid(), os.getppid(), "python -m stageload guard --busy ltx --")
    snapshots = iter([[own, (4242, 1, "python ltx-2-mlx/run.py")], [own]])
    last = [None]

    def processes():
        last[0] = next(snapshots, last[0])
        return last[0]

    io, logs = make_io(processes=processes)
    assert run_guarded(PASS, GuardConfig(busy=("ltx",), **QUICK), io) == 0
    assert any("busy: python ltx-2-mlx/run.py" in line for line in logs)


def test_a_swap_rise_over_the_budget_kills_the_command():
    io, logs = make_io(swaps=(1 * GB, 1 * GB, 2 * GB, 10 * GB))
    start = time.monotonic()
    code = run_guarded(SLEEP, GuardConfig(swap_budget=8 * GB, **QUICK), io)
    assert code == EXIT_KILLED
    assert time.monotonic() - start < 15
    assert any("killing" in line for line in logs)


def test_two_low_memory_samples_in_a_row_kill_the_command():
    io, _ = make_io(levels=(90, 5, 5))
    assert run_guarded(SLEEP, GuardConfig(**QUICK), io) == EXIT_KILLED


def test_the_command_exit_code_is_passed_through():
    io, _ = make_io()
    cmd = [sys.executable, "-c", "import sys; sys.exit(7)"]
    assert run_guarded(cmd, GuardConfig(**QUICK), io) == 7


def test_a_gap_between_two_busy_jobs_is_not_enough_to_start():
    busy_job = (4242, 1, "python ltx-2-mlx generate")
    snapshots = iter([[busy_job], [], [busy_job], [], [], [], [], []])
    last = [[]]

    def processes():
        last[0] = next(snapshots, last[0])
        return last[0]

    clock_at_start = []
    io, logs = make_io(processes=processes)
    original_log = io.log

    def log(line):
        if "guard: start" in line:
            clock_at_start.append(io.clock())
        original_log(line)

    io.log = log
    config = GuardConfig(busy=("ltx",), quiet_for=2.0, poll=1.0, grace=1.0)
    assert run_guarded(PASS, config, io) == 0
    assert clock_at_start and clock_at_start[0] >= 5.0
    assert any("quiet for" in line for line in logs)


def test_the_guard_and_every_shell_above_it_are_ignored():
    me, parent, grandparent = os.getpid(), 70001, 70002
    processes = [
        (me, parent, "python -m stageload guard --busy ltx -- job"),
        (parent, grandparent, "zsh -c 'nohup stageload guard --busy ltx -- job &'"),
        (grandparent, 1, "zsh -c 'source snapshot; eval stageload guard --busy ltx'"),
        (70003, 1, "python ltx-2-mlx/run.py"),
    ]
    assert ancestors(processes, me) == {me, parent, grandparent}
    io, logs = make_io(processes=lambda: processes[:3])
    assert run_guarded(PASS, GuardConfig(busy=("ltx",), **QUICK), io) == 0


def test_a_command_ended_by_a_signal_reports_128_plus_the_signal():
    io, _ = make_io()
    cmd = [sys.executable, "-c", "import os, signal; os.kill(os.getpid(), signal.SIGTERM)"]
    assert run_guarded(cmd, GuardConfig(**QUICK), io) == 128 + signal.SIGTERM


def test_a_command_that_cannot_start_is_reported_without_a_traceback():
    io, logs = make_io()
    code = run_guarded(["/nonexistent/stageload-test-binary"], GuardConfig(**QUICK), io)
    assert code == EXIT_CANNOT_START
    assert any("cannot start" in line for line in logs)


def _gone(pid, timeout=5.0):
    """True once ``pid`` no longer runs (a zombie waiting for its parent counts as gone)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                               text=True).stdout.strip()
        if not state or state.startswith("Z"):
            return True
        time.sleep(0.1)
    return False


def _wait_for_file(path, timeout=15.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists() and path.read_text().strip():
            return int(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path} was never written")


def test_a_terminated_guard_stops_its_command_first(tmp_path):
    pidfile = tmp_path / "child.pid"
    child = ("import os, pathlib, time; "
             f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)")
    guard = f"""
import sys
from stageload.guard import GuardConfig, GuardIO, run_guarded
io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [], log=lambda line: None)
sys.exit(run_guarded([sys.executable, "-c", {child!r}], GuardConfig(poll=0.1, grace=1.0), io))
"""
    proc = subprocess.Popen([sys.executable, "-c", guard])
    child_pid = _wait_for_file(pidfile)
    proc.send_signal(signal.SIGTERM)
    assert proc.wait(timeout=15) == 128 + signal.SIGTERM
    assert _gone(child_pid)


def test_stopping_a_group_also_kills_members_that_outlive_the_leader(tmp_path):
    pidfile = tmp_path / "member.pid"
    member = ("import os, pathlib, signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
              f"pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid())); time.sleep(60)")
    leader = ("import subprocess, sys, time; "
              f"subprocess.Popen([sys.executable, '-c', {member!r}]); time.sleep(60)")
    proc = subprocess.Popen([sys.executable, "-c", leader], start_new_session=True)
    member_pid = _wait_for_file(pidfile)
    _stop_group(proc, grace=1.0)
    assert proc.returncode is not None
    assert _gone(member_pid)
