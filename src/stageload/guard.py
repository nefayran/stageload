"""Start one heavy command only when there is room, and stop it before it takes the machine down.

Unlike a system-wide OOM killer, the guard needs no root, and the only processes it ever stops
are the command it started and that command's process group. If the guard itself gets SIGINT,
SIGTERM or SIGHUP, it stops the command first, unless it was started to ignore that signal (as
``nohup`` does for SIGHUP). SIGKILL cannot be caught: a command whose guard was killed that way
keeps running.
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from types import FrameType
from typing import Any

EXIT_KILLED = 3
EXIT_NO_ROOM = 4
EXIT_CANNOT_START = 127
GB = 2**30

Process = tuple[int, int, str]  # pid, parent pid, command line


@dataclass(frozen=True)
class GuardConfig:
    wait_free: int = 40
    swap_budget: int = 8 * GB
    swap_limit: int | None = None
    kill_free: int = 10
    busy: tuple[str, ...] = ()
    quiet_for: float = 0.0
    start_timeout: float = 1800.0
    poll: float = 5.0
    grace: float = 5.0


def _stderr(line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"{stamp} {line}", file=sys.stderr, flush=True)


@dataclass
class GuardIO:
    level: Callable[[], int]
    swap_used: Callable[[], int]
    processes: Callable[[], list[Process]]
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = _stderr


def list_processes() -> list[Process]:
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) == 3 else ""))
    return rows


def system_io() -> GuardIO:
    from .meter import memorystatus_level, swap_used

    return GuardIO(level=memorystatus_level, swap_used=swap_used, processes=list_processes)


def ancestors(processes: Iterable[Process], pid: int) -> set[int]:
    """``pid`` and every process above it. Their command lines contain the guard's own
    arguments, so they always match its ``--busy`` patterns."""
    parent = {p: pp for p, pp, _ in processes}
    chain = {pid}
    while pid in parent and parent[pid] not in chain and parent[pid] > 1:
        pid = parent[pid]
        chain.add(pid)
    return chain


def busy_processes(
    regexes: Sequence[re.Pattern[str]], processes: Iterable[Process], exclude: set[int]
) -> list[str]:
    """Command lines that match a pattern, like ``pgrep -f``, leaving out ``exclude``."""
    return [
        command
        for pid, _, command in processes
        if pid not in exclude and any(r.search(command) for r in regexes)
    ]


def _gb(nbytes: int) -> str:
    return f"{nbytes / GB:.1f} GB"


def _no_room_reasons(
    config: GuardConfig,
    io: GuardIO,
    regexes: Sequence[re.Pattern[str]],
    quiet: dict[str, float | None],
) -> list[str]:
    reasons = []
    level = io.level()
    swap = io.swap_used()
    if level < config.wait_free:
        reasons.append(f"memory available {level}% < {config.wait_free}%")
    if config.swap_limit is not None and swap + config.swap_budget > config.swap_limit:
        reasons.append(
            f"swap {_gb(swap)} + budget {_gb(config.swap_budget)} > limit {_gb(config.swap_limit)}"
        )
    if regexes:
        processes = io.processes()
        busy = busy_processes(regexes, processes, ancestors(processes, os.getpid()))
        if busy:
            quiet["since"] = None
            reasons.append("busy: " + "; ".join(command[:80] for command in busy[:3]))
        elif config.quiet_for > 0:
            # a chain of jobs leaves short gaps between them: on 2026-10-05 a run started in
            # such a gap, and the next video job took swap past its budget within 20 s
            if quiet["since"] is None:
                quiet["since"] = io.clock()
            quiet_so_far = io.clock() - quiet["since"]
            if quiet_so_far < config.quiet_for:
                reasons.append(f"quiet for {quiet_so_far:.0f} s of {config.quiet_for:.0f} s")
    return reasons


class _Signals:
    """While installed, SIGINT, SIGTERM and SIGHUP are recorded instead of acting at once, so the
    guard can stop its command whatever it was doing when one arrived. A signal the guard was
    started to ignore stays ignored. Handlers can only be installed from the main thread."""

    HANDLED = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)

    def __init__(self) -> None:
        self.first: int | None = None
        self.count = 0
        self._previous: dict[int, Any] = {}

    def __enter__(self) -> _Signals:
        if threading.current_thread() is threading.main_thread():
            for signum in self.HANDLED:
                if signal.getsignal(signum) is not signal.SIG_IGN:
                    self._previous[signum] = signal.signal(signum, self._record)
        return self

    def __exit__(self, *exc: object) -> None:
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)

    def _record(self, signum: int, frame: FrameType | None) -> None:
        if self.first is None:
            self.first = signum
        self.count += 1


_SLICE = 0.2  # seconds: how long the guard can take to notice a signal


def _nap(io: GuardIO, seconds: float, signals: _Signals | None) -> None:
    slices = max(1, math.ceil(seconds / _SLICE))
    for _ in range(slices):
        if signals is not None and signals.count:
            return
        io.sleep(seconds / slices)


def wait_for_room(config: GuardConfig, io: GuardIO, signals: _Signals | None = None) -> bool:
    """True once there is room; False after ``start_timeout``, or as soon as a signal arrives."""
    regexes = [re.compile(pattern) for pattern in config.busy]
    deadline = io.clock() + config.start_timeout
    last = ""
    quiet: dict[str, float | None] = {"since": None}
    while True:
        if signals is not None and signals.count:
            return False
        reasons = _no_room_reasons(config, io, regexes, quiet)
        if not reasons:
            return True
        reason = ", ".join(reasons)
        kind = reason.split(" for ")[0] if reason.startswith("quiet") else reason
        if kind != last:
            io.log(f"guard: waiting ({reason})")
            last = kind
        if io.clock() >= deadline:
            io.log(f"guard: no room after {config.start_timeout:.0f} s ({reason})")
            return False
        _nap(io, config.poll, signals)


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _stop_group(
    proc: subprocess.Popen[bytes], grace: float, signals: _Signals | None = None
) -> None:
    """SIGTERM to the command's group, then SIGKILL to whatever is left of it once ``grace``
    seconds have passed, or at once if the guard gets another signal meanwhile.

    The grace is for the whole group: a shell that exits at once does not cut short the cleanup
    of the job it started.
    """
    pgid = proc.pid
    seen = signals.count if signals is not None else 0
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        proc.poll()  # reap the leader, or it stays in the group as a zombie
        if not _group_alive(pgid):
            break
        if signals is not None and signals.count > seen:
            break
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _wait(proc: subprocess.Popen[bytes], seconds: float, signals: _Signals) -> int | None:
    """The command's exit code if it ends within ``seconds``; None if it is still running or a
    signal arrived."""
    deadline = time.monotonic() + seconds
    while True:
        remaining = deadline - time.monotonic()
        try:
            return proc.wait(timeout=max(0.0, min(_SLICE, remaining)))
        except subprocess.TimeoutExpired:
            if signals.count or remaining <= _SLICE:
                return None


def _exit_code(code: int) -> int:
    """A command ended by a signal reports -N; a shell reports 128+N."""
    return 128 - code if code < 0 else code


def _signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return f"signal {signum}"


def run_guarded(cmd: Sequence[str], config: GuardConfig, io: GuardIO | None = None) -> int:
    io = io or system_io()
    with _Signals() as signals:
        room = wait_for_room(config, io, signals)
        if signals.first is not None:
            io.log(f"guard: got {_signal_name(signals.first)} before starting the command")
            return 128 + signals.first
        if not room:
            return EXIT_NO_ROOM
        swap_at_start = io.swap_used()
        limit = swap_at_start + config.swap_budget
        if config.swap_limit is not None:
            limit = min(limit, config.swap_limit)
        io.log(
            f"guard: start, swap {_gb(swap_at_start)}, stop above {_gb(limit)} "
            f"or under {config.kill_free}% available twice"
        )
        try:
            proc = subprocess.Popen(list(cmd), start_new_session=True)
        except OSError as error:
            io.log(f"guard: cannot start {cmd[0]!r}: {error.strerror or error}")
            return EXIT_CANNOT_START
        try:
            return _watch(proc, config, io, signals, limit)
        except BaseException:
            # the command cannot be watched any more, so it does not keep running unwatched
            _stop_group(proc, config.grace, signals)
            raise


def _watch(
    proc: subprocess.Popen[bytes],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    limit: int,
) -> int:
    low = 0
    while True:
        if signals.first is not None:
            io.log(
                f"guard: got {_signal_name(signals.first)}, stopping pid {proc.pid} and its group"
            )
            _stop_group(proc, config.grace, signals)
            return 128 + signals.first
        code = _wait(proc, config.poll, signals)
        if code is not None:
            io.log(f"guard: exit {_exit_code(code)}")
            return _exit_code(code)
        if signals.first is not None:
            continue
        level = io.level()
        swap = io.swap_used()
        low = low + 1 if level < config.kill_free else 0
        if swap > limit or low >= 2:
            if swap > limit:
                why = f"swap {_gb(swap)} > {_gb(limit)}"
            else:
                why = f"memory available {level}% < {config.kill_free}% twice"
            io.log(f"guard: killing pid {proc.pid} and its group ({why})")
            _stop_group(proc, config.grace, signals)
            return EXIT_KILLED
