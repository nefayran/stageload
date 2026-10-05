"""Start one heavy command only when there is room, and stop it before it takes the machine down.

Unlike a system-wide OOM killer, the guard needs no root, and the only processes it ever stops
are the command it started and that command's process group. If the guard itself is stopped
with SIGINT, SIGTERM or SIGHUP, it stops the command first; SIGKILL cannot be caught, and a
command whose guard was killed that way keeps running.
"""

from __future__ import annotations

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


def wait_for_room(config: GuardConfig, io: GuardIO) -> bool:
    regexes = [re.compile(pattern) for pattern in config.busy]
    deadline = io.clock() + config.start_timeout
    last = ""
    quiet: dict[str, float | None] = {"since": None}
    while True:
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
        io.sleep(config.poll)


def _stop_group(proc: subprocess.Popen[bytes], grace: float) -> None:
    """SIGTERM to the command's group, then SIGKILL to whatever is left of it after ``grace``."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _exit_code(code: int) -> int:
    """A command ended by a signal reports -N; a shell reports 128+N."""
    return 128 - code if code < 0 else code


class _Signalled(BaseException):
    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _raise_signalled(signum: int, frame: FrameType | None) -> None:
    raise _Signalled(signum)


def run_guarded(cmd: Sequence[str], config: GuardConfig, io: GuardIO | None = None) -> int:
    io = io or system_io()
    if not wait_for_room(config, io):
        return EXIT_NO_ROOM
    swap_at_start = io.swap_used()
    limit = swap_at_start + config.swap_budget
    if config.swap_limit is not None:
        limit = min(limit, config.swap_limit)
    io.log(
        f"guard: start, swap {_gb(swap_at_start)}, stop above {_gb(limit)} "
        f"or under {config.kill_free}% available twice"
    )
    handled = (signal.SIGTERM, signal.SIGHUP)
    previous = {}
    if threading.current_thread() is threading.main_thread():
        previous = {signum: signal.signal(signum, _raise_signalled) for signum in handled}
    try:
        try:
            proc = subprocess.Popen(list(cmd), start_new_session=True)
        except OSError as error:
            io.log(f"guard: cannot start {cmd[0]!r}: {error.strerror or error}")
            return EXIT_CANNOT_START
        low = 0
        try:
            while True:
                try:
                    code = proc.wait(timeout=config.poll)
                except subprocess.TimeoutExpired:
                    code = None
                if code is not None:
                    io.log(f"guard: exit {_exit_code(code)}")
                    return _exit_code(code)
                level = io.level()
                swap = io.swap_used()
                low = low + 1 if level < config.kill_free else 0
                if swap > limit or low >= 2:
                    if swap > limit:
                        why = f"swap {_gb(swap)} > {_gb(limit)}"
                    else:
                        why = f"memory available {level}% < {config.kill_free}% twice"
                    io.log(f"guard: killing pid {proc.pid} and its group ({why})")
                    _stop_group(proc, config.grace)
                    return EXIT_KILLED
        except KeyboardInterrupt:
            io.log(f"guard: interrupted, stopping pid {proc.pid} and its group")
            _stop_group(proc, config.grace)
            return 128 + signal.SIGINT
        except _Signalled as stopped:
            io.log(f"guard: got signal {stopped.signum}, stopping pid {proc.pid} and its group")
            _stop_group(proc, config.grace)
            return 128 + stopped.signum
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
