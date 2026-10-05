"""Start one heavy command only when there is room, and stop it before it takes the machine down.

Unlike a system-wide OOM killer, the guard needs no root, and the only processes it ever stops
are the command it started and that command's process group.
"""

from __future__ import annotations

import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone

EXIT_KILLED = 3
EXIT_NO_ROOM = 4
GB = 2**30


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
    processes: Callable[[], list[tuple[int, str]]]
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = _stderr


def list_processes() -> list[tuple[int, str]]:
    out = subprocess.run(
        ["ps", "-axo", "pid=,command="], capture_output=True, text=True, check=True
    ).stdout
    rows = []
    for line in out.splitlines():
        pid, _, command = line.strip().partition(" ")
        if pid.isdigit():
            rows.append((int(pid), command.strip()))
    return rows


def system_io() -> GuardIO:
    from .meter import memorystatus_level, swap_used

    return GuardIO(level=memorystatus_level, swap_used=swap_used, processes=list_processes)


def busy_processes(
    patterns: Iterable[str], processes: Iterable[tuple[int, str]], exclude: set[int]
) -> list[str]:
    """Command lines that match a pattern. ``exclude`` holds the guard and its parent shell,
    whose own command lines contain the patterns."""
    regexes = [re.compile(p) for p in patterns]
    return [
        command
        for pid, command in processes
        if pid not in exclude and any(r.search(command) for r in regexes)
    ]


def _gb(nbytes: int) -> str:
    return f"{nbytes / GB:.1f} GB"


def _no_room_reasons(
    config: GuardConfig, io: GuardIO, exclude: set[int], quiet: dict[str, float | None]
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
    if config.busy:
        busy = busy_processes(config.busy, io.processes(), exclude)
        if busy:
            quiet["since"] = None
            reasons.append("busy: " + "; ".join(command[:80] for command in busy[:3]))
        elif config.quiet_for > 0:
            # a chain of jobs leaves short gaps between them: 05.10 a run started in such a gap
            # and the next video job took swap past its budget within 20 s
            if quiet["since"] is None:
                quiet["since"] = io.clock()
            quiet_so_far = io.clock() - quiet["since"]
            if quiet_so_far < config.quiet_for:
                reasons.append(
                    f"quiet for {quiet_so_far:.0f} s of {config.quiet_for:.0f} s"
                )
    return reasons


def wait_for_room(config: GuardConfig, io: GuardIO) -> bool:
    exclude = {os.getpid(), os.getppid()}
    deadline = io.clock() + config.start_timeout
    last = ""
    quiet: dict[str, float | None] = {"since": None}
    while True:
        reasons = _no_room_reasons(config, io, exclude, quiet)
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


def _stop_group(proc: subprocess.Popen, grace: float) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


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
    proc = subprocess.Popen(list(cmd), start_new_session=True)
    low = 0
    try:
        while True:
            try:
                code = proc.wait(timeout=config.poll)
            except subprocess.TimeoutExpired:
                code = None
            if code is not None:
                io.log(f"guard: exit {code}")
                return code
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
        _stop_group(proc, config.grace)
        return 130
