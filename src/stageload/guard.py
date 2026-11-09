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
from collections import deque
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import FrameType
from typing import Any

EXIT_KILLED = 3
EXIT_NO_ROOM = 4
EXIT_CANNOT_START = 127
GB = 2**30

Process = tuple[int, int, str]  # pid, parent pid, command line
Resident = tuple[int, int, str]  # pid, resident bytes, command line

PAUSE_MAX_S = 600.0
# the kill floor of a --gpu command, which is never paused: its --kill-free is at least this
GPU_KILL_LEVEL = 15
# 2026-10-30: eleven GPU video renders in a week were stopped for swap 7.1-9.6 GB over their start while memory available
# never fell under 25 %; a --gpu command is never paused, so swap past its budget stops it only under this
# level or over this much swap machine-wide (a replay of 257 render traces: 0 stops instead of 18)


def env_number(name: str, default: float) -> float:
    """A number from the environment, or ``default`` when it is unset or not a number."""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_gb(name: str, default: float) -> Callable[[], int]:
    return lambda: int(env_number(name, default) * GB)


GPU_SWAP_LEVEL = env_number("STAGELOAD_GPU_SWAP_LEVEL", 25)
GPU_SWAP_CEILING = _env_gb("STAGELOAD_GPU_SWAP_CEILING_GB", 24)


def _env_level(name: str, default: int) -> Callable[[], int]:
    return lambda: int(env_number(name, default))


def pause_max_default() -> float:
    """$STAGELOAD_PAUSE_MAX_S: how long a command paused for outside swap may stay paused before
    it is stopped as before (default 600; 0 turns pausing off)."""
    try:
        return max(0.0, float(os.environ["STAGELOAD_PAUSE_MAX_S"]))
    except (KeyError, ValueError):
        return PAUSE_MAX_S


@dataclass(frozen=True)
class GuardConfig:
    wait_free: int = 35
    swap_budget: int = 8 * GB
    swap_limit: int | None = None
    kill_free: int = 10
    busy: tuple[str, ...] = ()
    quiet_for: float = 0.0
    start_timeout: float = 1800.0
    poll: float = 5.0
    grace: float = 5.0
    pause_max: float = field(default_factory=pause_max_default)
    # absolute readings, whatever the swap was at the start (2026-10-24: the Mac hung with swap
    # 8.3 GB before a GPU video render even began, the compressor at 15 GB, the level at 40 %).
    # `stageload run` starts nothing over the gate in any lane
    gate_swap: int = field(default_factory=_env_gb("STAGELOAD_GATE_SWAP_GB", 6))
    gate_compressor: int = field(default_factory=_env_gb("STAGELOAD_GATE_COMPRESSOR_GB", 8))
    gate_level: int = field(default_factory=_env_level("STAGELOAD_GATE_LEVEL", 35))
    # 2026-10-27: at this level or over, with the compressor under its gate, swap over
    # the gate is pages macOS never read back (8.4-9.2 GB for hours at 84-88 % available) and holds no start
    gate_stale_swap_level: int = field(default_factory=_env_level("STAGELOAD_GATE_STALE_SWAP_LEVEL", 35))
    # with this much memory available the compressor, like swap, holds pages nobody touches (idle
    # waiters and apps): up to this size it does not hold a start (29.10: 47 % available, 9.5 GB
    # compressed, swap 11.5 GB, and nothing could start)
    gate_stale_compressor: int = field(default_factory=_env_gb("STAGELOAD_GATE_STALE_COMPRESSOR_GB", 24))
    # over these, a running command whose own footprint is flat is paused until the gate clears
    pause_swap: int = field(default_factory=_env_gb("STAGELOAD_PAUSE_SWAP_GB", 9))
    pause_compressor: int = field(default_factory=_env_gb("STAGELOAD_PAUSE_COMPRESSOR_GB", 12))
    pause_level: int = field(default_factory=_env_level("STAGELOAD_PAUSE_LEVEL", 25))
    # what the budget leaves to the OS, the apps and the security agents. 2026-10-24: 23.4 + 7.7 GB
    # of two queue jobs plus a 12 GB reserve fit 48 GB, and the compressor still reached 21.9 GB
    reserve: int = field(default_factory=_env_gb("STAGELOAD_RESERVE_GB", 16))
    # False for a command that runs on another machine (a remote lane): this machine's memory
    # readings never pause or stop it, since pausing it frees nothing here
    watch_memory: bool = True
    # False for a --gpu command: it is never paused (2026-10-24: a paused GPU video render had its Metal buffers
    # unwired, compressed and swapped, 20 GB wired fell to 3 GB and swap reached 12.5 GB, so no
    # resume could come before the kill). It is stopped at once at the kill floor instead
    pausable: bool = True


def _stderr(line: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%SZ")
    print(f"{stamp} {line}", file=sys.stderr, flush=True)


@dataclass(frozen=True)
class Growth:
    """One running guarded job as its trace shows it: how much memory it added to the machine."""

    who: str
    grown: int  # footprint added since the job started: peak of the last minute less first sample
    recent: int  # footprint added over the last minute
    me: bool = False  # the job this guard watches
    rise: int = 0  # the largest rise of its footprint within the last minute, trough to peak
    started: float = 0.0  # when it started (a timestamp; 0: not known)
    paused: bool = False  # its guard has paused it
    gpu: bool = False  # a --gpu job: never paused, so never the one the blame falls on


@dataclass
class GuardIO:
    level: Callable[[], int]
    swap_used: Callable[[], int]
    processes: Callable[[], list[Process]]
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic
    log: Callable[[str], None] = _stderr
    # the running guarded jobs, this guard's own among them; None: the guard knows of no others
    jobs: Callable[[], list[Growth]] | None = None
    # footprint of a pid and every process under it; None: the guard cannot tell whether swap
    # growth is its command's, and stops it for swap as before
    footprint: Callable[[int], int] | None = None
    # the processes outside the queue that hold the most resident memory, largest first
    neighbours: Callable[[], list[Resident]] | None = None
    # told "paused" or "running" when the guard pauses or resumes its command
    on_state: Callable[[str], None] | None = None
    # bytes the memory compressor occupies, and wired memory; None: not known
    compressor: Callable[[], int] | None = None
    wired: Callable[[], int] | None = None
    # the footprint of every process outside the guarded jobs, summed; None: not known
    outside: Callable[[], int] | None = None
    # the processes with the largest footprint, queue and not, largest first
    top: Callable[[], list[Resident]] | None = None


def list_processes() -> list[Process]:
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,command="], capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",  # 2026-10-16: ps cuts a long command line inside a UTF-8 character
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 2)
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            rows.append((int(parts[0]), int(parts[1]), parts[2] if len(parts) == 3 else ""))
    return rows


def job_footprint(pid: int) -> int:
    from .queue import tree_footprint

    return tree_footprint(pid)


def top_resident(roots: set[int] | None = None, count: int = 5) -> list[Resident]:
    """The processes that hold the most resident memory, leaving out ``roots`` (by default this
    guard) and every process under them: what the machine runs besides the guarded jobs."""
    roots = {os.getpid()} if roots is None else roots
    out = subprocess.run(
        ["ps", "-axo", "pid=,ppid=,rss=,command="], capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout
    rows, children = [], {}
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) < 3 or not all(part.isdigit() for part in parts[:3]):
            continue
        pid, ppid, rss = int(parts[0]), int(parts[1]), int(parts[2]) * 1024
        rows.append((pid, rss, parts[3] if len(parts) == 4 else ""))
        children.setdefault(ppid, []).append(pid)
    skip, todo = set(), list(roots)
    while todo:
        pid = todo.pop()
        if pid not in skip:
            skip.add(pid)
            todo.extend(children.get(pid, []))
    rows = [row for row in rows if row[0] not in skip]
    return sorted(rows, key=lambda row: row[1], reverse=True)[:count]


def list_footprints() -> list[tuple[int, int, int, str]]:
    """(pid, parent pid, footprint, command line) of every process whose footprint can be read."""
    from .meter import process_footprint

    rows = []
    for pid, ppid, command in list_processes():
        try:
            rows.append((pid, ppid, process_footprint(pid), command))
        except OSError:
            continue  # gone, or another user's
    return rows


def _tree(rows: Iterable[tuple[int, int, int, str]], roots: set[int]) -> set[int]:
    children: dict[int, list[int]] = {}
    for pid, ppid, _, _ in rows:
        children.setdefault(ppid, []).append(pid)
    seen, todo = set(), list(roots)
    while todo:
        pid = todo.pop()
        if pid not in seen:
            seen.add(pid)
            todo.extend(children.get(pid, []))
    return seen


def outside_footprint(roots: set[int] | None = None) -> int:
    """The footprint of every process outside ``roots`` (by default this guard) and what they
    started: what the machine runs besides the guarded jobs."""
    rows = list_footprints()
    skip = _tree(rows, {os.getpid()} if roots is None else roots)
    return sum(fp for pid, _, fp, _ in rows if pid not in skip)


def top_footprint(count: int = 5) -> list[Resident]:
    rows = sorted(list_footprints(), key=lambda row: row[2], reverse=True)[:count]
    return [(pid, fp, command) for pid, _, fp, command in rows]


def _process_name(command: str) -> str:
    """The program, and the script it runs when it is an interpreter: «python build_short.py»."""
    words = command.split()
    while words and os.path.basename(words[0]) == "nice":
        words = words[3:] if words[1:2] == ["-n"] else words[1:]
    if not words:
        return "?"
    name = os.path.basename(words[0])
    if re.match(r"(?:python[\d.]*|zsh|bash|sh|node)$", name):
        rest = [w for w in words[1:] if not w.startswith("-") and not w.lstrip("-").isdigit()]
        if rest:
            name += " " + os.path.basename(rest[0])
    return name[:60]


def _top_text(io: GuardIO) -> str:
    if io.top is None:
        return "not known"
    try:
        rows = io.top()
    except Exception as error:  # noqa: BLE001 - the listing only names the grower
        return f"not known ({type(error).__name__}: {error})"
    return "; ".join(f"pid {pid} {_process_name(command)} {_gb(fp)}"
                     for pid, fp, command in rows) or "none"


def system_io() -> GuardIO:
    from .meter import compressor_used, memorystatus_level, swap_used, wired_used

    mac = sys.platform == "darwin"
    return GuardIO(level=memorystatus_level, swap_used=swap_used, processes=list_processes,
                   footprint=job_footprint, neighbours=top_resident,
                   compressor=compressor_used if mac else None, wired=wired_used if mac else None,
                   outside=outside_footprint if mac else None, top=top_footprint if mac else None)


def read_optional(reader: Callable[[], int] | None) -> int | None:
    """A reading that may be missing: None when there is no reader or it fails."""
    if reader is None:
        return None
    try:
        return int(reader())
    except Exception:  # noqa: BLE001 - an unknown reading is left out of the checks
        return None


def over_gate(
    swap: int, compressor: int | None, level: int, *, swap_max: int, compressor_max: int,
    level_min: int,
) -> list[str]:
    """What is past an absolute threshold, as words; empty when nothing is."""
    reasons = []
    if swap > swap_max:
        reasons.append(f"swap {_gb(swap)} > {_gb(swap_max)}")
    if compressor is not None and compressor > compressor_max:
        reasons.append(f"compressor {_gb(compressor)} > {_gb(compressor_max)}")
    if level < level_min:
        reasons.append(f"memory available {level}% < {level_min}%")
    return reasons


def start_gate(config: GuardConfig, swap: int, compressor: int | None, level: int) -> list[str]:
    return over_gate(swap, compressor, level, swap_max=config.gate_swap,
                     compressor_max=config.gate_compressor, level_min=config.gate_level)


def pause_gate(config: GuardConfig, swap: int, compressor: int | None, level: int) -> list[str]:
    return over_gate(swap, compressor, level, swap_max=config.pause_swap,
                     compressor_max=config.pause_compressor, level_min=config.pause_level)


def gate_text(config: GuardConfig) -> str:
    return (f"swap {_gb(config.gate_swap)}, compressor {_gb(config.gate_compressor)}, "
            f"available {config.gate_level}%")


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
    idle: Callable[[], bool] | None = None,
) -> list[str]:
    reasons = []
    level = io.level()
    swap = io.swap_used()
    if level < config.wait_free:
        reasons.append(f"memory available {level}% < {config.wait_free}%")
    # 2026-10-26: the run's own --swap-limit held a long job on stale swap while the Mac was idle,
    # and 67 jobs behind it; on an idle Mac it does not count
    if (config.swap_limit is not None and swap + config.swap_budget > config.swap_limit
            and not (idle is not None and idle())):
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


def wait_for_room(config: GuardConfig, io: GuardIO, signals: _Signals | None = None,
                  idle: Callable[[], bool] | None = None) -> bool:
    """True once there is room; False after ``start_timeout``, or as soon as a signal arrives."""
    regexes = [re.compile(pattern) for pattern in config.busy]
    deadline = io.clock() + config.start_timeout
    last = ""
    quiet: dict[str, float | None] = {"since": None}
    while True:
        if signals is not None and signals.count:
            return False
        reasons = _no_room_reasons(config, io, regexes, quiet, idle)
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


def group_members(pgid: int) -> list[int]:
    """The processes of a group that can still take a signal: every member but the zombies."""
    out = subprocess.run(
        ["ps", "-axo", "pid=,pgid=,stat="], capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout
    members = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 3 or not parts[0].isdigit() or parts[2].startswith("Z"):
            continue
        if parts[1] == str(pgid):
            members.append(int(parts[0]))
    return members


def park_loops_in(pgid: int) -> set[int]:
    """The members of a group that are a parked ticket's wait (queue._PARK_LOOP) or its children: the loop shell, its
    ``sleep`` and its ``--near`` check. 31.10.2026: a job whose command runs ``stageload run`` inside it parks that inner
    ticket in the job's own process group; a SIGSTOP of the job then froze the parked check for 5 h and the ram lane
    with it (drama-cn-V10j, its check pid 45322 in state T)."""
    from .queue import PARK_MARK
    out = subprocess.run(
        ["ps", "-axww", "-o", "pid=,ppid=,pgid=,command="], capture_output=True, text=True, check=True,
        encoding="utf-8", errors="replace",
    ).stdout
    rows = []
    for line in out.splitlines():
        parts = line.split(None, 3)
        if len(parts) >= 3 and parts[0].isdigit() and parts[2] == str(pgid):
            rows.append((int(parts[0]), int(parts[1]), parts[3] if len(parts) > 3 else ""))
    loops = {pid for pid, _pp, cmd in rows if PARK_MARK in cmd}
    return loops | {pid for pid, ppid, _cmd in rows if ppid in loops}


def _pause_group(pgid: int) -> bool:
    """SIGSTOP to every member of the group except parked tickets' waits (park_loops_in); False once the group is
    gone. Without parked waits in the group this is the plain group signal."""
    try:
        skip = park_loops_in(pgid)
    except (OSError, subprocess.SubprocessError):
        skip = set()
    if not skip:
        return _signal_group(pgid, signal.SIGSTOP)
    try:
        pids = group_members(pgid)
    except (OSError, subprocess.SubprocessError):
        return True
    for pid in pids:
        if pid in skip:
            continue
        try:
            os.kill(pid, signal.SIGSTOP)
        except OSError:
            pass
    return bool(pids)


def _signal_group(
    pgid: int, signum: int, members: Callable[[int], list[int]] = group_members
) -> bool:
    """``signum`` to every process of the group; False once the group is gone.

    On macOS ``killpg`` fails with EPERM, not ESRCH, while the group still exists but none of its
    members can take a signal: each has exited and is not reaped yet, still freeing its memory or
    already a zombie. A render that held 28 GB, much of it in swap, takes longer than the grace to
    free it, and on 2026-10-21 that EPERM ended two guards with a traceback. Each
    live member is then signalled on its own, and a member that refuses is skipped.
    """
    try:
        os.killpg(pgid, signum)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    try:
        pids = members(pgid)
    except (OSError, subprocess.SubprocessError):
        return True
    for pid in pids:
        try:
            os.kill(pid, signum)
        except OSError:
            pass  # gone meanwhile, or not ours to signal
    return bool(pids) if signum == 0 else True


def _group_alive(pgid: int, members: Callable[[int], list[int]] = group_members) -> bool:
    return _signal_group(pgid, 0, members)


def _stop_group(
    proc: subprocess.Popen[bytes],
    grace: float,
    signals: _Signals | None = None,
    members: Callable[[int], list[int]] = group_members,
) -> None:
    """SIGTERM to the command's group, then SIGKILL to whatever is left of it once ``grace``
    seconds have passed, or at once if the guard gets another signal meanwhile.

    The grace is for the whole group: a shell that exits at once does not cut short the cleanup
    of the job it started. It never raises: a guard that dies here leaves its job unreaped and
    its run without a result. The leader is always reaped.
    """
    pgid = proc.pid
    seen = signals.count if signals is not None else 0
    try:
        if _signal_group(pgid, signal.SIGTERM, members):
            # a paused group takes SIGTERM only once it runs again; to a running one this is a no-op
            _signal_group(pgid, signal.SIGCONT, members)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline:
                proc.poll()  # reap the leader, or it stays in the group as a zombie
                if not _group_alive(pgid, members):
                    break
                if signals is not None and signals.count > seen:
                    break
                time.sleep(0.05)
            _signal_group(pgid, signal.SIGKILL, members)
    except Exception as error:  # noqa: BLE001 - whatever failed, the leader still has to go
        _stderr(f"guard: stopping group {pgid}: {type(error).__name__}: {error}")
        try:
            proc.kill()
        except OSError:
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
            return _watch(proc, config, io, signals, limit, swap_at_start)
        except BaseException:
            # the command cannot be watched any more, so it does not keep running unwatched
            _stop_group(proc, config.grace, signals)
            raise


def blame(jobs: Sequence[Growth]) -> tuple[float, Growth | None]:
    """Our job's share of the memory the running guarded jobs added, and the job that added more
    than ours did, if one did: that one explains the swap growth, ours does not.

    Swap grows because the jobs together hold more than the machine has, so each job is charged
    the part of the growth its own footprint explains. On 2026-10-21 light jobs of 3-4 GB were
    killed twelve times for swap that a 29 GB render beside them added. Alone, or not seen by the
    others (no trace), a guard owns all of the growth, as before.
    """
    me = next((job for job in jobs if job.me), None)
    others = [job for job in jobs if not job.me]
    if me is None or not others:
        return 1.0, None
    mine = max(0, me.grown)
    total = mine + sum(max(0, job.grown) for job in others)
    if total <= 0:
        return 1.0, None
    top = max(others, key=lambda job: job.grown)
    return mine / total, (top if top.grown > mine else None)


def critical_rank(jobs: Sequence[Growth]) -> int:
    """How many running guarded jobs stop before ours when memory runs out: the one that grew most
    over the last minute goes first, then the one that added most memory."""
    me = next((job for job in jobs if job.me), None)
    if me is None:
        return 0
    key = (me.recent, me.grown)
    return sum(1 for job in jobs if not job.me and (job.recent, job.grown) > key)


_CRITICAL_SAMPLES = 2  # samples under --kill-free before the first job is stopped
# each further job waits two more: a 28 GB render takes seconds to give its memory back
_CRITICAL_STEP = 2


def _read_jobs(io: GuardIO) -> list[Growth] | None:
    if io.jobs is None:
        return None
    try:
        return io.jobs()
    except Exception as error:  # noqa: BLE001 - not knowing the others means acting alone
        io.log(f"guard: cannot read the other jobs ({type(error).__name__}: {error})")
        return None


# swap over the limit is outside pressure when our command's footprint grew by no more than this
# over the last FLAT_WINDOW_S: 512 MB, or a quarter of the swap growth over the same time. On
# 2026-10-23 the two GPU video renders a Next.js dev server killed moved 0.3 GB at most in 10 minutes
FLAT_WINDOW_S = 60.0
FLAT_SLACK = 512 * 2**20
FLAT_SHARE = 0.25
# a paused command resumes once swap is this far under its limit (at most half the budget)
RESUME_MARGIN = GB
# polls in a row the trigger must hold before a command is paused
PAUSE_STREAK = 2
# a compressor reading that was dealt with (a pause resumed, or another queue job blamed) counts
# again only once the compressor grows this much more, or falls back under the start gate: the
# pages of a paused command stay compressed after it resumes, and would pause it again at once
COMPRESSOR_REARM = 2 * GB


@dataclass(frozen=True)
class _Pressure:
    footprint: int  # our command's footprint now
    grew: int  # its growth over the window
    swap_grew: int  # swap growth over the window
    span: float  # seconds the window covers


def _outside_pressure(samples: deque[tuple[float, int, int]]) -> _Pressure | None:
    """The window's readings when our command's footprint stayed flat while swap went up; None
    when it grew, or when there are too few samples to tell."""
    if len(samples) < 2:
        return None
    (t0, swap0, fp0), (t1, swap1, fp1) = samples[0], samples[-1]
    pressure = _Pressure(footprint=fp1, grew=fp1 - fp0, swap_grew=swap1 - swap0, span=t1 - t0)
    flat = pressure.grew <= max(FLAT_SLACK, FLAT_SHARE * max(0, pressure.swap_grew))
    return pressure if flat else None


def window_rise(values: Sequence[int]) -> int:
    """The largest rise within ``values``, from a trough to a later peak: what a job added over
    the window even if it gave some of it back since."""
    rise, low = 0, None
    for value in values:
        low = value if low is None else min(low, value)
        rise = max(rise, value - low)
    return rise


def _flat_slack(samples: deque[tuple[float, int, int]]) -> int:
    swap_grew = samples[-1][1] - samples[0][1] if len(samples) >= 2 else 0
    return int(max(FLAT_SLACK, FLAT_SHARE * max(0, swap_grew)))


def queue_culprit(
    me_rise: int | None, jobs: Sequence[Growth] | None, slack: int
) -> tuple[str, Growth | None, list[Growth]]:
    """Whose growth the pressure is, among the running queue jobs, over the last minute.

    Returns (verdict, target, growing): "alone" when no other queue job is seen (the guard acts as
    before); "outside" when no queue job rose by more than ``slack``; else the target is the most
    recently started job among those that rose, and the verdict is "ours" when it is our job,
    "theirs" when it is another. A job counts as grown when its rise passes ``slack``.

    2026-10-24: a 23.4 GB audio job (ram) and a 7.7 GB 4K build (light) ran side by side and the
    compressor reached 14.5 GB. The audio job's guard saw its own footprint flat (+0.2 GB in 61 s),
    called the pressure outside and paused its command, while the build beside it had risen 1 GB;
    the processes outside the queue held under 0.5 GB each.
    """
    others = [job for job in jobs or () if not job.me]
    if not others:
        return "alone", None, []
    listed = next((job for job in jobs or () if job.me), None)
    rise = max(listed.rise if listed is not None else 0, me_rise or 0)
    me = Growth(who=listed.who if listed is not None else "ours", grown=0, recent=0, me=True,
                rise=rise, started=listed.started if listed is not None else float("inf"))
    growing = [job for job in (me, *others) if job.rise > slack]
    if not growing:
        return "outside", None, []
    # a --gpu job is never paused: the newest of the others that rose is, if one did
    pool = [job for job in growing if not job.gpu] or growing
    target = max(pool, key=lambda job: (job.started, job.rise))
    return ("ours" if target.me else "theirs"), target, growing


def _growing_text(growing: Sequence[Growth]) -> str:
    return ", ".join(f"{'ours' if job.me else job.who} {job.rise / GB:+.1f} GB"
                     + (" (paused)" if job.paused else "") for job in growing)


def _read_footprint(io: GuardIO, pid: int) -> int | None:
    if io.footprint is None:
        return None
    try:
        return int(io.footprint(pid))
    except Exception:  # noqa: BLE001 - an unknown footprint means stopping for swap as before
        return None


def _neighbours_text(io: GuardIO) -> str:
    if io.neighbours is None:
        return "not known"
    try:
        rows = io.neighbours()
    except Exception as error:  # noqa: BLE001 - the listing only names the neighbour
        return f"not known ({type(error).__name__}: {error})"
    return "; ".join(f"pid {pid} {_gb(rss)} {command[:70]}" for pid, rss, command in rows) or "none"


def _tell(io: GuardIO, state: str) -> None:
    if io.on_state is None:
        return
    try:
        io.on_state(state)
    except Exception as error:  # noqa: BLE001 - the queue display must not stop the guard
        io.log(f"guard: cannot note the state {state!r} ({type(error).__name__}: {error})")


def _watch(
    proc: subprocess.Popen[bytes],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    limit: int,
    swap_at_start: int | None = None,
) -> int:
    """Wait for the command, and stop it when swap passes ``limit`` or memory runs out.

    With ``io.jobs`` the guard knows the other running guarded jobs. Swap growth that their
    footprints explain raises our limit instead of counting against our budget, and a job that
    added less memory than another is not stopped for swap at all. When memory runs out, the job
    that grew most recently stops first and every other job waits two samples more per job ahead
    of it. Without ``io.jobs``, or alone, nothing changes: swap over the limit or two samples
    under ``--kill-free`` stop the command.

    With ``io.footprint``, swap over the limit while the command's own footprint stayed flat over
    the last minute is pressure from outside the queue (2026-10-23: a Next.js dev server of 3-4 GB
    killed two GPU video renders, 45 minutes of work, and kept running). The command's group is paused
    with SIGSTOP instead of stopped, and resumed with SIGCONT once swap is ``RESUME_MARGIN`` under
    the limit. It is stopped as before when it stays paused for ``config.pause_max`` seconds, or
    when memory runs out meanwhile.

    The same pause holds on absolute readings, whatever the swap was at the start: swap over
    ``config.pause_swap``, the compressor over ``config.pause_compressor`` or the level under
    ``config.pause_level`` while the command's footprint stayed flat. Such a pause ends once swap
    and the level are back within the start gate; the compressor is not waited for, since the
    paused command's own pages fill it (2026-10-24: the Mac hung at swap 8.3 GB + a GPU video render,
    the compressor at 15 GB, with the swap budget, counted from the start, far away).

    Before a trip is called outside pressure, the growth of every running queue job over the last
    minute is compared (``queue_culprit``). When some queue job rose, the most recently started of
    them is the one paused: its own guard pauses it, and the others run on. Only when no queue job
    rose is the pressure outside. A command alone in the queue is judged as before.
    """
    low = 0
    prev = swap_at_start
    spared_for: str | None = None
    waiting_rank = 0
    samples: deque[tuple[float, int, int]] = deque()  # (clock, swap, our footprint)
    paused_at: float | None = None
    paused_total = 0.0
    margin = min(RESUME_MARGIN, config.swap_budget // 2)
    compressor_seen: int | None = None  # a compressor reading already dealt with
    swap_seen: int | None = None  # swap at the last resume: our own pages, already judged
    outside_at_pause: int | None = None
    blamed: str | None = None  # the queue job last named as the culprit
    ran_on = False  # a --gpu command told it runs on under this pressure
    gpu_swap_told = False  # a --gpu command told it runs on past its swap budget
    streak = 0  # polls in a row with swap over the limit or the pause gate tripped
    while True:
        if signals.first is not None:
            io.log(
                f"guard: got {_signal_name(signals.first)}, stopping pid {proc.pid} and its group"
            )
            _stop_group(proc, config.grace, signals)
            return 128 + signals.first
        code = _wait(proc, config.poll, signals)
        if code is not None:
            if paused_at is not None:
                paused_total += io.clock() - paused_at
            note = f" (paused {paused_total:.0f} s in all)" if paused_total else ""
            io.log(f"guard: exit {_exit_code(code)}{note}")
            return _exit_code(code)
        if signals.first is not None:
            continue
        if not config.watch_memory:
            continue  # 2026-10-24: an arch-lane watcher of 0.0 GB was paused and killed here
        level = io.level()
        swap = io.swap_used()
        compressor = read_optional(io.compressor)
        if compressor is not None and compressor_seen is not None and (
                compressor <= config.gate_compressor):
            compressor_seen = None
        fresh = compressor
        if compressor is not None and compressor_seen is not None and (
                compressor <= compressor_seen + COMPRESSOR_REARM):
            fresh = None  # dealt with already
        if swap_seen is not None and swap <= config.gate_swap:
            swap_seen = None
        new_swap = swap_seen is None or swap > swap_seen + margin
        hot = pause_gate(config, swap if new_swap else 0, fresh, level)
        now = io.clock()
        if paused_at is None:
            footprint = _read_footprint(io, proc.pid)
            if footprint is not None:
                samples.append((now, swap, footprint))
                while len(samples) > 1 and samples[1][0] <= now - FLAT_WINDOW_S:
                    samples.popleft()
        low = low + 1 if level < config.kill_free else 0
        grew = 0 if prev is None else swap - prev
        prev = swap
        ceiling = config.swap_limit is not None and swap > config.swap_limit and new_swap
        jobs = None
        if io.jobs is not None and (grew or low or swap > limit or ceiling or hot):
            jobs = _read_jobs(io)
        share, cause = blame(jobs) if jobs else (1.0, None)
        before = limit
        if grew and share < 1.0:
            limit += int(grew * (1.0 - share))
        over = swap > limit or ceiling
        if (over and not ceiling and not config.pausable and level >= GPU_SWAP_LEVEL
                and swap <= GPU_SWAP_CEILING()):
            if not gpu_swap_told:
                io.log(
                    f"guard: swap {_gb(swap)} is past {_gb(limit)}, but memory available is "
                    f"{level}% >= {GPU_SWAP_LEVEL:.0f}% and swap is under "
                    f"{_gb(GPU_SWAP_CEILING())}: a --gpu command runs on"
                )
                gpu_swap_told = True
            over = False
        spared = swap > before and not over
        if over and cause is not None:
            over, spared = False, True
        if spared:
            name = cause.who if cause is not None else "the other jobs"
            if name != spared_for:
                mine = next((job.grown for job in jobs or () if job.me), 0)
                theirs = cause.grown if cause is not None else sum(
                    job.grown for job in jobs or () if not job.me
                )
                io.log(
                    f"guard: swap {_gb(swap)} is past our {_gb(before)}, but {name} added "
                    f"{_gb(theirs)} against our {_gb(mine)}: not ours to stop "
                    f"(our limit now {_gb(limit)})"
                )
                spared_for = name
        elif swap <= before:
            spared_for = None
        rank = critical_rank(jobs) if jobs and low >= _CRITICAL_SAMPLES else 0
        critical = low >= _CRITICAL_SAMPLES + _CRITICAL_STEP * rank
        streak = streak + 1 if (over or hot) else 0
        if low >= _CRITICAL_SAMPLES and not critical and rank != waiting_rank:
            io.log(
                f"guard: memory available {level}% < {config.kill_free}%, "
                f"{rank} job(s) that grew more recently stop first"
            )
        waiting_rank = rank if low >= _CRITICAL_SAMPLES else 0
        if paused_at is not None and not critical:
            paused_for = now - paused_at
            # swap is not asked: the paused command's own pages go there while it waits
            outside = read_optional(io.outside) if outside_at_pause is not None else None
            calm = []
            # 2026-10-29: resumed at the start gate's level (35 %), a light job paused beside a GPU
            # video render never saw it and was stopped after pause_max; the pause floor is the measure
            if level >= config.pause_level:
                calm.append(f"memory available {level}% >= {config.pause_level}%")
            if outside is not None and outside <= outside_at_pause - RESUME_MARGIN:
                calm.append(f"the footprint outside the queue fell to {_gb(outside)} from "
                            f"{_gb(outside_at_pause)}")
            if calm:
                _signal_group(proc.pid, signal.SIGCONT)
                paused_total += paused_for
                paused_at = None  # the window spans the pause: a paused command did not grow
                if compressor is not None and compressor > config.gate_compressor:
                    compressor_seen = compressor  # our own pages, compressed while we waited
                swap_seen = swap
                limit = max(limit, swap + margin)
                io.log(
                    f"guard: resuming pid {proc.pid} and its group (SIGCONT): "
                    f"{', '.join(calm)} after {paused_for:.0f} s paused"
                )
                _tell(io, "running")
                continue
            if paused_for < config.pause_max:
                continue
            over = True  # stopped as before
        elif hot and not over and not critical and not config.pausable:
            if not ran_on:
                io.log(
                    f"guard: {', '.join(f'{reason} machine-wide' for reason in hot)}, but a "
                    f"--gpu command is never paused: it runs on, and stops at once under "
                    f"{config.kill_free}% available or over {_gb(limit)} of swap. Largest "
                    f"footprints: {_top_text(io)}"
                )
                ran_on = True
        elif ((over or hot) and not critical and config.pause_max > 0 and config.pausable
              and streak < PAUSE_STREAK):
            # one reading is not a trend (29.10: light jobs were paused on the poll their ffmpeg
            # started in); the next poll decides
            over = False
        elif (over or hot) and not critical and config.pause_max > 0 and config.pausable:
            pressure = _outside_pressure(samples)
            mine = window_rise([fp for _, _, fp in samples]) if len(samples) >= 2 else None
            verdict, target, growing = queue_culprit(mine, jobs, _flat_slack(samples))
            trigger = [f"swap {_gb(swap)} > {_gb(limit)}"] if over else []
            trigger += [f"{reason} machine-wide" for reason in hot]
            if verdict == "theirs" and target is not None:
                # another queue job rose: its own guard pauses it, ours runs on
                if target.who != blamed:
                    io.log(
                        f"guard: {', '.join(trigger)}, but of the queue jobs that rose over the "
                        f"last {FLAT_WINDOW_S:.0f} s ({_growing_text(growing)}) {target.who} "
                        f"started last: the pressure is its, not ours to pause or stop. Largest "
                        f"footprints: {_top_text(io)}"
                    )
                    blamed = target.who
                over = False
                if fresh is not None and fresh > config.pause_compressor:
                    compressor_seen = fresh
            elif verdict == "ours" or pressure is not None:
                outside_now = read_optional(io.outside)
                if _pause_group(proc.pid):
                    paused_at = now
                    outside_at_pause = outside_now
                    until = [f"memory available is {config.pause_level}% or more"]
                    if outside_now is not None:
                        until.append(f"the footprint outside the queue, {_gb(outside_now)} now, "
                                     f"falls {_gb(RESUME_MARGIN)}")
                    if verdict == "ours":
                        why = (f"and of the queue jobs that rose over the last "
                               f"{FLAT_WINDOW_S:.0f} s ({_growing_text(growing)}) ours started "
                               f"last: the pressure is ours")
                    else:
                        assert pressure is not None
                        flat = (f", and no queue job rose more than {_gb(_flat_slack(samples))}"
                                if verdict == "outside" else "")
                        why = (f"but our footprint {_gb(pressure.footprint)} changed by "
                               f"{pressure.grew / GB:+.1f} GB over the last {pressure.span:.0f} s "
                               f"while swap grew {pressure.swap_grew / GB:+.1f} GB{flat}: the "
                               f"pressure is from outside the queue. Largest resident outside "
                               f"the queue: {_neighbours_text(io)}")
                    io.log(
                        f"guard: pausing pid {proc.pid} and its group (SIGSTOP): "
                        f"{', '.join(trigger)}, {why}. Largest footprints: {_top_text(io)}. "
                        f"Resuming once {' or '.join(until)}, stopping after "
                        f"{config.pause_max:.0f} s paused"
                    )
                    _tell(io, "paused")
                    continue
        elif not (over or hot):
            blamed = None
            ran_on = False
        if over or critical:
            if paused_at is not None and not critical:
                left = [f"still memory available {level}% < {config.pause_level}%"]
                outside = read_optional(io.outside) if outside_at_pause is not None else None
                if outside is not None:
                    left.append(f"the footprint outside the queue {_gb(outside)} against "
                                f"{_gb(outside_at_pause)} at the pause")
                why = f"paused {now - paused_at:.0f} s, {', '.join(left)}"
            elif over:
                why = f"swap {_gb(swap)} > {_gb(limit)}"
            else:
                why = f"memory available {level}% < {config.kill_free}% {low} times"
                if low == _CRITICAL_SAMPLES:
                    why = f"memory available {level}% < {config.kill_free}% twice"
            io.log(f"guard: killing pid {proc.pid} and its group ({why}). Largest footprints: "
                   f"{_top_text(io)}")
            _stop_group(proc, config.grace, signals)
            return EXIT_KILLED
