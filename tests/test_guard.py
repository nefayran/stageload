import os
import sys
import time

from stageload.guard import EXIT_KILLED, EXIT_NO_ROOM, GuardConfig, GuardIO, run_guarded

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
    own = (os.getpid(), "python -m stageload guard --busy ltx --")
    snapshots = iter([[own, (4242, "python ltx-2-mlx/run.py")], [own]])
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
