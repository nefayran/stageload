"""One queue for the whole machine: heavy commands start one at a time, in the order they asked.

``stageload guard`` decides alone: every guard polls memory and the process list on its own, so
two guards that find room at the same moment both start, and a guard can start in the gap
between two jobs of another session's chain. ``stageload run`` puts each command in a lane
under ``STAGELOAD_HOME`` (``~/.stageload`` by default). A command starts when no older entry of
its lane is still alive and it holds the lane's lock; then the guard's own memory checks run as
usual. The command inherits the lock, so the lane stays taken while it runs even if the process
that queued it is killed with SIGKILL.

Every run is metered: the footprint of the command and every process under it, swap and the
memory level go to a trace, and one line per finished run goes to ``history.jsonl``.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from .guard import (
    EXIT_CANNOT_START,
    EXIT_NO_ROOM,
    GuardConfig,
    GuardIO,
    Process,
    _gb,
    _nap,
    _signal_name,
    _Signals,
    _stop_group,
    _watch,
    list_processes,
    system_io,
    wait_for_room,
)
from .meter import MemoryMeter, Readers, process_footprint

DEFAULT_LANE = "ram"
_TICKET = re.compile(r"^\d{20}-\d+\.json$")


def default_home() -> Path:
    return Path(os.environ.get("STAGELOAD_HOME") or Path.home() / ".stageload")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Readers never see half a file: the entry is written aside, then renamed into place."""
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _live_tickets(lane_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Entries of a lane, oldest first. An entry whose process is gone is removed: a run killed
    with SIGKILL cannot remove its own."""
    live = []
    for path in sorted(p for p in lane_dir.iterdir() if _TICKET.match(p.name)):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = int(data.get("pid") or path.stem.split("-")[1])
        if _alive(pid):
            live.append((path, data))
        else:
            path.unlink(missing_ok=True)
    return live


def load_busy(home: Path) -> tuple[str, ...]:
    """Patterns from ``busy.txt``, one per line: every run checks the same list, so a heavy
    engine added there holds the starts of all sessions at once."""
    path = home / "busy.txt"
    if not path.exists():
        return ()
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            re.compile(line)
        except re.error:
            continue
        patterns.append(line)
    return tuple(patterns)


def tree_footprint(root: int, processes: Callable[[], list[Process]] = list_processes) -> int:
    """Footprint of ``root`` and every process under it: a shell script's own footprint says
    nothing about the engine it started."""
    children: dict[int, list[int]] = {}
    for pid, ppid, _ in processes():
        children.setdefault(ppid, []).append(pid)
    total, todo, seen = 0, [root], set()
    while todo:
        pid = todo.pop()
        if pid in seen:
            continue
        seen.add(pid)
        todo.extend(children.get(pid, []))
        try:
            total += process_footprint(pid)
        except OSError:
            pass  # it ended between the listing and the read
    return total


def list_queue(home: Path | None = None, lane: str | None = None) -> list[dict[str, Any]]:
    home = Path(home) if home is not None else default_home()
    root = home / "queue"
    if not root.exists():
        return []
    lanes = [root / lane] if lane else sorted(p for p in root.iterdir() if p.is_dir())
    rows = []
    for lane_dir in lanes:
        if not lane_dir.exists():
            continue
        for position, (_, data) in enumerate(_live_tickets(lane_dir)):
            rows.append(
                {
                    "lane": lane_dir.name,
                    "position": position,
                    "state": data.get("state", "waiting"),
                    "who": data.get("who", ""),
                    "pid": data.get("pid"),
                    "since": data.get("started") or data.get("since"),
                    "cmd": data.get("cmd", ""),
                }
            )
    return rows


def default_trace(home: Path, who: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", who) or "job"
    return home / "traces" / f"{stamp}-{name}-{os.getpid()}.jsonl"


def _wait_turn(
    ticket: Path,
    lane_dir: Path,
    lock: IO[str],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
) -> bool:
    """True once every older entry of the lane is gone and the lane's lock is ours."""
    deadline = io.clock() + config.start_timeout
    last = ""
    while True:
        if signals.count:
            return False
        ahead = [data for path, data in _live_tickets(lane_dir) if path.name < ticket.name]
        if not ahead:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                reason = "the lane is still held by a command whose run is gone"
        else:
            names = ", ".join(str(d.get("who") or d.get("pid")) for d in ahead[:3])
            reason = f"{len(ahead)} ahead: {names}"
        if reason != last:
            io.log(f"queue: waiting in {lane_dir.name} ({reason})")
            last = reason
        if io.clock() >= deadline:
            io.log(f"queue: no turn after {config.start_timeout:.0f} s ({reason})")
            return False
        _nap(io, config.poll, signals)


def run_queued(
    cmd: Sequence[str],
    config: GuardConfig,
    *,
    io: GuardIO | None = None,
    home: Path | None = None,
    lane: str = DEFAULT_LANE,
    who: str = "",
    trace: Path | None = None,
) -> int:
    io = io or system_io()
    home = Path(home) if home is not None else default_home()
    lane_dir = home / "queue" / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    config = replace(config, busy=tuple(config.busy) + load_busy(home))
    ticket = lane_dir / f"{time.time_ns():020d}-{os.getpid()}.json"
    entry = {"pid": os.getpid(), "who": who, "lane": lane, "cmd": " ".join(cmd)[:300],
             "since": _utc(), "state": "waiting"}
    _write_json(ticket, entry)
    queued_at = io.clock()
    lock = (lane_dir / "run.lock").open("a")
    room = False
    try:
        with _Signals() as signals:
            turn = _wait_turn(ticket, lane_dir, lock, config, io, signals)
            if turn:
                room = wait_for_room(config, io, signals)
            if signals.first is not None:
                io.log(f"queue: got {_signal_name(signals.first)} before starting the command")
                return 128 + signals.first
            if not turn or not room:
                return EXIT_NO_ROOM
            entry.update(state="running", started=_utc())
            _write_json(ticket, entry)
            waited = io.clock() - queued_at
            return _run(cmd, config, io, signals, lock, home, entry, trace, waited)
    finally:
        ticket.unlink(missing_ok=True)
        lock.close()


def _run(
    cmd: Sequence[str],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    lock: IO[str],
    home: Path,
    entry: dict[str, Any],
    trace: Path | None,
    waited: float,
) -> int:
    swap_at_start = io.swap_used()
    limit = swap_at_start + config.swap_budget
    if config.swap_limit is not None:
        limit = min(limit, config.swap_limit)
    io.log(
        f"queue: start in {entry['lane']} after {waited:.0f} s, swap {_gb(swap_at_start)}, "
        f"stop above {_gb(limit)} or under {config.kill_free}% available twice"
    )
    try:
        # the command keeps the lane's lock open: the lane stays taken while it runs
        proc = subprocess.Popen(list(cmd), start_new_session=True, pass_fds=(lock.fileno(),))
    except OSError as error:
        io.log(f"queue: cannot start {cmd[0]!r}: {error.strerror or error}")
        return EXIT_CANNOT_START
    started = io.clock()
    meter = None
    if trace is not None:
        readers = Readers(footprint=lambda: tree_footprint(proc.pid), swap_used=io.swap_used,
                          level=io.level)
        meta = {"who": entry["who"], "lane": entry["lane"], "cmd": entry["cmd"]}
        meter = MemoryMeter(trace, meta=meta, readers=readers).__enter__()
    try:
        code = _watch(proc, config, io, signals, limit)
    except BaseException:
        _stop_group(proc, config.grace, signals)
        raise
    finally:
        if meter is not None:
            meter.__exit__(None, None, None)
    summary = meter.summary() if meter is not None else {}
    swap = summary.get("swap", {})
    record = {
        "end": _utc(),
        "lane": entry["lane"],
        "who": entry["who"],
        "cmd": entry["cmd"],
        "exit": code,
        "waited_s": round(waited, 1),
        "run_s": round(io.clock() - started, 1),
        "peak_footprint": summary.get("peak_footprint"),
        "swap_start": swap.get("start"),
        "swap_peak": swap.get("peak"),
        "swap_end": swap.get("end"),
        "min_level": summary.get("min_level"),
        "trace": str(trace) if trace is not None else None,
    }
    with (home / "history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(record) + "\n")
    peak = record["peak_footprint"]
    io.log(
        f"queue: exit {code} after {record['run_s']:.0f} s"
        + (f", peak {_gb(peak)}" if peak else "")
        + (f", trace {trace}" if trace is not None else "")
    )
    return code
