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
import fnmatch
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any

from .guard import (
    EXIT_CANNOT_START,
    EXIT_KILLED,
    EXIT_NO_ROOM,
    GPU_KILL_LEVEL,
    Growth,
    GuardConfig,
    GuardIO,
    Process,
    _gb,
    _group_alive,
    _nap,
    _signal_group,
    _signal_name,
    _Signals,
    _stop_group,
    _watch,
    gate_text,
    list_processes,
    outside_footprint,
    over_gate,
    read_optional,
    _no_room_reasons,
    start_gate,
    system_io,
    top_resident,
    window_rise,
)
from .meter import MemoryMeter, Readers, machine_ram, process_footprint

DEFAULT_LANE = "ram"
FAR_POLL_S = 30.0
_HISTORY_CACHE: dict[tuple[Path, int, int], list[dict | None]] = {}
_TICKET_CACHE: dict[Path, tuple[int, int, dict[str, Any]]] = {}
# a ticket in these states has its lane's turn: running, paused by its guard for swap from outside
# the queue, or holding the turn while it waits for the GPU or for memory
_RUNS = ("running", "paused")
_HOLDS_LANE = (*_RUNS, "gpu-wait", "mem-wait")
GB = 2**30
# Predict the peak from the command when --mem-peak is not given. Process footprints miss much of
# Metal memory, so known engines are matched here.
PEAK_PATTERNS: tuple[tuple[re.Pattern[str], int], ...] = (
    (re.compile(r"local-llm/h3-forks/[^/\s]+/h3\b|h3_exec[\w-]*\.sh|local-llm/h3(?:\.c|-lab)/h3\b"),
     25 * GB),
    (re.compile(r"ltx-2-mlx"), 43 * GB),
    (re.compile(r"(?:^|[/\s])[Bb]lender\b.*\s(?:-b|--background)\b"), 4 * GB),
    (re.compile(r"realcugan|\brife"), 3 * GB),
    (re.compile(r"mmaudio|foley", re.IGNORECASE), 8 * GB),
    (re.compile(r"whisper", re.IGNORECASE), 3 * GB),
)
HISTORY_FACTOR = 1.3  # over the largest peak the history has for the same --who
DEFAULT_PEAK = 4 * GB
# A ram-lane command with no matching engine or history entry: heavy jobs go to ram. An MMAudio
# job counted at the 4 GB default once peaked at 23.4 GB.
RAM_LANE_PEAK = 16 * GB
# "grew recently"; also how long a stopped job keeps its blame while it frees its memory
RECENT_S = 60.0
_STALE_S = 30.0  # a trace without a sample for this long says nothing: its meter is not running
_TRACE_TAIL = 128 * 1024  # bytes read from the end of a trace: some 800 s of samples
# when a run ends, its owner keeps the lane's next turn this long for a command sent after it.
# 29.10: no hold after a run that ended well (the next step of a pipeline takes its turn as
# everyone), only after a quick failure, long enough to fix the command
HOLD_S = 0.0
QUICK_FAIL_S = 10.0  # a command that fails sooner than this was most likely given wrong arguments
QUICK_FAIL_HOLD_S = 30.0  # the hold after such a failure, long enough to fix the command
_TICKET = re.compile(r"^\d{20}-\d+\.json$")
# a guard or run that has not started its command yet: its command line names the engine it
# will start, but it is not running that engine
_WRAPPER = re.compile(
    r"(?:-m\s+stageload|(?:^|[/\s])stageload)\s+(?:guard|run)\b|-m\s+stageload\.queue\b"
)


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


# Fields another program writes into a waiting entry (a queue planner, a priority list): the run keeps its entry in
# memory and rewrites it while it waits, so these are taken from the file each time instead of being lost.
KEPT_FIELDS = ("deadline", "planned_by", "planned_why", "real_deadline", "expect_s", "slot", "risk")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Readers never see half a file: the entry is written aside, then renamed into place. A queue entry keeps the
    KEPT_FIELDS that are in its file now."""
    if path.name.startswith("0") and path.suffix == ".json" and path.exists():
        try:
            on_disk = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            on_disk = {}
        for key in KEPT_FIELDS:
            if key in on_disk:
                data[key] = on_disk[key]
    tmp = path.with_name(f".tmp-{path.name}")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    os.replace(tmp, path)


def _live_tickets(lane_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Entries of a lane, oldest first. An entry whose process is gone is removed: a run killed
    with SIGKILL cannot remove its own."""
    live = []
    paths = {p for p in lane_dir.iterdir() if _TICKET.match(p.name)}
    for path in list(_TICKET_CACHE):
        if path.parent == lane_dir and path not in paths:
            del _TICKET_CACHE[path]
    for path in sorted(paths):
        try:
            stat = path.stat()
            cached = _TICKET_CACHE.get(path)
            if cached is None or cached[:2] != (stat.st_mtime_ns, stat.st_size):
                cached = (stat.st_mtime_ns, stat.st_size,
                          json.loads(path.read_text(encoding="utf-8")))
                _TICKET_CACHE[path] = cached
            data = dict(cached[2])
        except (OSError, ValueError):
            _TICKET_CACHE.pop(path, None)
            continue
        pid = int(data.get("pid") or path.stem.split("-")[1])
        if _alive(pid):
            live.append((path, data))
        else:
            _resume_orphan(data)
            path.unlink(missing_ok=True)
    return live


def _resume_orphan(data: dict[str, Any]) -> None:
    """A run killed with SIGKILL while its guard had paused the command would leave it stopped
    for good, holding its lane's lock: it runs on unwatched, as any command whose guard was
    killed so."""
    if data.get("state") != "paused" or not data.get("pgid"):
        return
    try:
        os.killpg(int(data["pgid"]), signal.SIGCONT)
    except (OSError, ValueError, TypeError):
        pass


def _holds_lane(data: dict[str, Any]) -> bool:
    return data.get("state") in _HOLDS_LANE


# A waiting entry whose turn has come but that cannot start (memory, budget, GPU) says so in its
# ticket, and the entries behind it may go first while the mark is fresh. After BLOCKED_YIELD_S of
# yielding it keeps its place again, so a large job is not starved by a stream of small ones. An
# entry held by the GPU yields for as long as the GPU is taken: nothing it waits for is a matter
# of its place. An entry that gave its slot up while it waits (``released``) is never a holder.
BLOCKED_FRESH_S = 30.0
BLOCKED_YIELD_S = 20 * 60.0
BLOCKED_REFRESH_S = 10.0  # a blocked mark is rewritten at most this often
_BLOCKED_KEYS = ("blocked", "blocked_at", "blocked_since", "blocked_gpu", "released")


def _yields_turn(data: dict[str, Any], now: float | None = None) -> bool:
    now = time.time() if now is None else now
    at, since = data.get("blocked_at"), data.get("blocked_since")
    if not data.get("blocked") or not isinstance(at, (int, float)):
        return False
    fresh = now - at < BLOCKED_FRESH_S
    if data.get("blocked_gpu"):
        return fresh
    return fresh and (not isinstance(since, (int, float)) or now - since < BLOCKED_YIELD_S)


def _mark_blocked(
    ticket: Path, data: dict[str, Any], reason: str, now: float, *, gpu: bool = False,
    released: bool = False,
) -> None:
    """The ticket says the entry has its turn but cannot start, so the entries behind it may go.
    Rewritten only when the reason changes or every BLOCKED_REFRESH_S."""
    at = data.get("blocked_at")
    if (data.get("blocked") == reason and isinstance(at, (int, float))
            and now - at < BLOCKED_REFRESH_S and bool(data.get("released")) == released):
        return
    data.update(blocked=reason, blocked_at=now)
    data.setdefault("blocked_since", now)
    if gpu:
        data["blocked_gpu"] = True
    if released:
        data["released"] = True
    else:
        data.pop("released", None)
    _write_json(ticket, data)


def _clear_blocked(ticket: Path, data: dict[str, Any]) -> None:
    if any(key in data for key in _BLOCKED_KEYS):
        for key in _BLOCKED_KEYS:
            data.pop(key, None)
        _write_json(ticket, data)




def owner_of(who: str, owner: str | None = None) -> str:
    """Whose turn a run takes: --owner, else the part of --who before the first dash."""
    return owner or (who.split("-", 1)[0] if who else "") or "job"


def _history_tail(path: Path, limit: int) -> list[dict | None]:
    """Cache decoded history by file metadata, retaining bad lines for line-count slicing."""
    try:
        path = path.resolve()
        stat = path.stat()
        key = (path, stat.st_mtime_ns, stat.st_size)
        if key not in _HISTORY_CACHE:
            records = []
            for line in path.read_text(encoding="utf-8").splitlines()[-5000:]:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    records.append(None)
            for old in list(_HISTORY_CACHE):
                if old[0] == path:
                    del _HISTORY_CACHE[old]
            _HISTORY_CACHE[key] = records
        return _HISTORY_CACHE[key][-limit:]
    except OSError:
        return []


def _history_starts(lane_dir: Path, limit: int = 2000) -> dict[str, int]:
    """When each owner last started in this lane, by the finished runs in history.jsonl: runs
    queued by an older stageload, or before owners.json existed, still count."""
    path = lane_dir.parent.parent / "history.jsonl"
    starts: dict[str, int] = {}
    for record in _history_tail(path, limit):
        if record is None:
            continue
        try:
            if record.get("lane") != lane_dir.name or record.get("state") == "waiting":
                continue  # a run stopped while it waited never took a turn
            end = datetime.fromisoformat(record["end"]).timestamp()
            start = int((end - float(record.get("run_s") or 0)) * 1e9)
        except (ValueError, KeyError, TypeError):
            continue
        owner = owner_of(record.get("who", ""))
        starts[owner] = max(starts.get(owner, 0), start)
    return starts


def _last_starts(lane_dir: Path) -> dict[str, int]:
    starts = _history_starts(lane_dir)
    try:
        noted = json.loads((lane_dir / "owners.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        noted = {}
    for owner, start in noted.items():
        starts[owner] = max(starts.get(owner, 0), int(start))
    return starts


def _note_start(lane_dir: Path, owner: str) -> None:
    starts = _last_starts(lane_dir)
    starts[owner] = time.time_ns()
    _write_json(lane_dir / "owners.json", starts)


def _deadline(data: dict[str, Any]) -> float:
    """The entry's deadline as a timestamp, or infinity: an entry without one waits behind every
    entry with one."""
    value = data.get("deadline")
    if not value:
        return float("inf")
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return float("inf")


def fair_order(tickets: list[tuple[Path, dict[str, Any]]], last: dict[str, int]) -> list[Path]:
    """The order the waiting entries will start in. An entry with an earlier deadline
    (--deadline, the time what it makes is due) starts before every entry with a later one or
    none. Among entries of one deadline the owner who started least recently goes next, and each
    owner's own entries keep the order they joined in, so one owner's batch takes turns with
    other owners instead of holding the lane."""
    waiting = [(p, d) for p, d in tickets if not _holds_lane(d)]
    running = [(p, d) for p, d in tickets if _holds_lane(d)]
    order: list[Path] = []
    clock = dict(last)
    shares = load_shares(_home_of(tickets))
    used = _share_use(tickets, shares) if shares else None
    for due in sorted({_deadline(d) for _, d in waiting}):
        group = running + [(p, d) for p, d in waiting if _deadline(d) == due]
        order += _turns(group, clock, shares, used)
    return order


# --- owner shares ---
# ~/.stageload/shares (STAGELOAD_SHARES overrides it): group:glob|glob=share on the owner name;
# other=N takes every owner no glob matches. Within one deadline the next turn goes to the group
# with the least run time over the last 24 h (history.jsonl plus the running jobs) per unit of
# share;
# inside a group the owners keep their usual turns. Deadlines stay
# first. Without shares the queue is as before.
SHARE_WINDOW_S = 24 * 3600
SHARE_GUESS_S = 900.0     # a job's expected run time when its owner has no history in the window


def load_shares(home: Path | None) -> list[tuple[str, list[str], float]]:
    text = os.environ.get("STAGELOAD_SHARES")
    if text is None and home is not None:
        try:
            text = (home / "shares").read_text(encoding="utf-8")
        except OSError:
            text = ""
    out: list[tuple[str, list[str], float]] = []
    for part in re.split(r"[,\n]", text or ""):
        part = part.strip()
        if not part or part.startswith("#") or "=" not in part:
            continue
        name, _, value = part.rpartition("=")
        group, _, globs = name.partition(":")
        try:
            share = float(value)
        except ValueError:
            continue
        if share > 0:
            out.append(
                (
                    group.strip(),
                    [g.strip() for g in (globs or group).split("|") if g.strip()],
                    share,
                )
            )
    return out


def share_group(owner: str, shares: list[tuple[str, list[str], float]]) -> tuple[str, float]:
    """The share group of an owner and its share; an owner no glob matches goes to «other»,
    else it is its own group with the smallest share named."""
    for group, globs, share in shares:
        if group != "other" and any(fnmatch.fnmatchcase(owner, g) for g in globs):
            return group, share
    for group, _globs, share in shares:
        if group == "other":
            return "other", share
    return owner, min(s for _, _, s in shares)


def share_report(home: Path, lane: str = "ram") -> list[dict[str, Any]]:
    """Each share group's share and its use of the lane over the last 24 h, for
    `stageload queue --shares`."""
    shares = load_shares(home)
    if not shares:
        return []
    lane_dir = home / "queue" / lane
    tickets = _live_tickets(lane_dir) if lane_dir.exists() else []
    used = _share_use(tickets or [(lane_dir / "none", {})], shares)
    total_share = sum(s for _, _, s in shares)
    total_used = sum(v for k, v in used.items() if k != "_typical") or 1.0
    return [
        {
            "group": g,
            "share": round(100 * s / total_share, 1),
            "used_h": round(used.get(g, 0.0) / 3600, 2),
            "used": round(100 * used.get(g, 0.0) / total_used, 1),
        }
        for g, _, s in shares
    ]


def _home_of(tickets: list[tuple[Path, dict[str, Any]]]) -> Path | None:
    return tickets[0][0].parent.parent.parent if tickets else None


def _share_use(
    tickets: list[tuple[Path, dict[str, Any]]],
    shares: list[tuple[str, list[str], float]],
    now: float | None = None,
) -> dict[str, float]:
    """Seconds each share group ran in the tickets' lanes over the last 24 h: finished runs from
    history.jsonl (the part inside the window) and the running jobs so far. «_typical» maps an
    owner to its median run time, for the guesses."""
    now = time.time() if now is None else now
    lanes = {p.parent.name for p, _ in tickets}
    used: dict[str, float] = {}
    runs: dict[str, list[float]] = {}
    groups: dict[str, tuple[str, float]] = {}

    def group_for(owner: str) -> tuple[str, float]:
        if owner not in groups:
            groups[owner] = share_group(owner, shares)
        return groups[owner]

    home = _home_of(tickets)
    records = _history_tail(home / "history.jsonl", 5000) if home else []
    for r in records:
        if r is None:
            continue
        try:
            if r.get("lane") not in lanes or r.get("state") == "waiting":
                continue
            end = datetime.fromisoformat(r["end"]).timestamp()
            run = float(r.get("run_s") or 0)
        except (ValueError, KeyError, TypeError):
            continue
        owner = owner_of(r.get("who", ""))
        runs.setdefault(owner, []).append(run)
        part = min(end, now) - max(end - run, now - SHARE_WINDOW_S)
        if part > 0:
            g, _ = group_for(owner)
            used[g] = used.get(g, 0.0) + part
    for _, d in tickets:
        if d.get("state") != "running" or not d.get("started"):
            continue
        try:
            started = datetime.fromisoformat(d["started"]).timestamp()
        except (TypeError, ValueError):
            continue
        g, _ = group_for(_owner(d))
        used[g] = used.get(g, 0.0) + max(0.0, now - max(started, now - SHARE_WINDOW_S))
    used["_typical"] = {o: sorted(v)[len(v) // 2] for o, v in runs.items()}  # type: ignore[assignment]
    return used


def _turns(
    tickets: list[tuple[Path, dict[str, Any]]],
    clock: dict[str, int],
    shares: list[tuple[str, list[str], float]] | None = None,
    used: dict[str, Any] | None = None,
) -> list[Path]:
    """fair_order within one deadline; `clock` carries each owner's last turn on to the next
    deadline. With shares, the group furthest below its share goes next, and its pick adds the
    owner's typical run time to the group's use."""
    queues: dict[str, list[Path]] = {}
    for path, data in tickets:
        owner = data.get("owner") or owner_of(data.get("who", ""))
        if _holds_lane(data):
            # the running job is its owner's last start, even before the history has it
            try:
                when = data.get("started") or data["turn"]  # "turn": holding it for the GPU
                started = int(datetime.fromisoformat(when).timestamp() * 1e9)
            except (KeyError, TypeError, ValueError):
                started = time.time_ns()
            clock[owner] = max(clock.get(owner, 0), started)
            continue
        queues.setdefault(owner, []).append(path)
    tick = max(clock.values(), default=0)
    order: list[Path] = []
    while queues:
        if shares:
            typical = (used or {}).get("_typical", {})
            def need(o: str) -> tuple[float, int, str]:
                g, share = share_group(o, shares)
                return ((used or {}).get(g, 0.0) / share, clock.get(o, 0), queues[o][0].name)
            owner = min(queues, key=need)
            g, _ = share_group(owner, shares)
            if used is not None:
                used[g] = used.get(g, 0.0) + float(typical.get(owner, SHARE_GUESS_S))
        else:
            owner = min(queues, key=lambda o: (clock.get(o, 0), queues[o][0].name))
        order.append(queues[owner].pop(0))
        if not queues[owner]:
            del queues[owner]
        tick += 1
        clock[owner] = tick
    return order


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


def _without_wrappers(
    processes: Callable[[], list[Process]], home: Path
) -> Callable[[], list[Process]]:
    """The process list without the runs and guards that are waiting: each one's command line
    names the engine it will start, so the head of the queue would wait for the runs behind it
    and none would ever start. What a waiting entry started still counts; what a running entry
    started does not (the budget counts it at its predicted peak, and as busy it would keep a
    second ram run from ever starting beside the first)."""

    def listing() -> list[Process]:
        root = home / "queue"
        wrappers, runs = set(), set()
        if root.exists():
            for lane_dir in root.iterdir():
                if lane_dir.is_dir():
                    for _, d in _live_tickets(lane_dir):
                        wrappers.add(int(d["pid"]))
                        if d.get("state") in _RUNS:
                            runs.add(int(d["pid"]))
        rows = processes()
        children: dict[int, list[int]] = {}
        for pid, ppid, _ in rows:
            children.setdefault(ppid, []).append(pid)
        todo = list(runs)
        while todo:
            pid = todo.pop()
            for child in children.get(pid, []):
                if child not in wrappers:
                    wrappers.add(child)
                    todo.append(child)
        return [p for p in rows if p[0] not in wrappers and not _WRAPPER.search(p[2])]

    return listing


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
        live = _live_tickets(lane_dir)
        by_path = dict(live)
        running = [(p, d) for p, d in live if _holds_lane(d)]
        ordered = running + [(p, by_path[p]) for p in fair_order(live, _last_starts(lane_dir))]
        for position, (_, data) in enumerate(ordered):
            rows.append(
                {
                    "lane": lane_dir.name,
                    "position": position,
                    "state": data.get("state", "waiting"),
                    "who": data.get("who", ""),
                    "what": data.get("what") or "",
                    "owner": data.get("owner") or owner_of(data.get("who", "")),
                    "pid": data.get("pid"),
                    "gpu": bool(data.get("gpu")),
                    "since": data.get("started") or data.get("turn") or data.get("since"),
                    "paused_since": data.get("paused_since"),
                    "mem_peak": data.get("mem_peak"),
                    "mem_wait": data.get("mem_wait"),
                    "cmd": data.get("cmd", ""),
                }
            )
    return rows


def default_trace(home: Path, who: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", who) or "job"
    return home / "traces" / f"{stamp}-{name}-{os.getpid()}.jsonl"


def _active_hold(lane_dir: Path) -> dict[str, Any] | None:
    try:
        hold = json.loads((lane_dir / "hold.json").read_text(encoding="utf-8"))
        return hold if float(hold["until"]) > time.time() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _hold_lane(lane_dir: Path, owner: str, who: str, seconds: float) -> None:
    """One hold per finished run: the next one replaces it, and the first command its owner sends
    after the end uses it up."""
    _write_json(lane_dir / "hold.json", {"owner": owner, "who": who, "ended_ns": time.time_ns(),
                                         "until": time.time() + seconds})


def _hold_seconds(quick_failure: bool) -> float:
    """$STAGELOAD_HOLD_S overrides the hold after a run (0: no hold), for tests and lanes of
    unrelated one-off jobs."""
    try:
        return float(os.environ["STAGELOAD_HOLD_S"])
    except (KeyError, ValueError):
        return QUICK_FAIL_HOLD_S if quick_failure else HOLD_S


def _quick_failure(code: int, seconds: float) -> bool:
    """A run that failed at once, most likely on its own arguments, and not one stopped by the
    guard or by a signal."""
    stopped = code in (EXIT_KILLED, EXIT_NO_ROOM) or code >= 128
    return code != 0 and not stopped and seconds < QUICK_FAIL_S


def _owner(data: dict[str, Any]) -> str:
    return data.get("owner") or owner_of(data.get("who", ""))


def held_for(ticket: Path, data: dict[str, Any], hold: dict[str, Any] | None) -> bool:
    """True when the hold is this entry's to use: its owner's, and sent after the run that set it
    ended. Entries the owner queued before (a batch) take their turns as usual, or one owner's
    batch would keep the lane from everyone."""
    if hold is None or _owner(data) != hold["owner"]:
        return False
    try:
        return int(ticket.name.split("-", 1)[0]) > int(hold["ended_ns"])
    except (KeyError, TypeError, ValueError):
        return False


def lane_ahead(
    ticket: Path,
    live: list[tuple[Path, dict[str, Any]]],
    last: dict[str, int],
    hold: dict[str, Any] | None,
    slots: int = 1,
) -> tuple[list[dict[str, Any]], str]:
    """The entries that go before ``ticket`` in its lane, and why it waits ("" when it does not).

    ``slots`` runs of the lane may hold it at once: an entry at the head of the waiting order
    goes while fewer than that hold the lane.

    When a run ends, its owner keeps the lane's next turn for a short while (``hold``): the next
    step of a pipeline, or the command fixed after a failure, goes before the entries of other
    owners instead of losing the lane to a long-running job. An entry with a deadline still goes
    in its deadline's order, hold or not.
    """
    by_path = dict(live)
    order = fair_order(live, last)
    now = time.time()
    mine = by_path.get(ticket, {})
    # a GPU entry at the head of the lane says in its ticket that the GPU is taken: while it does,
    # no waiting GPU entry can start, and an entry without --gpu passes them all
    gpu_taken = any(d.get("blocked_gpu") and _yields_turn(d, now) for _, d in live)

    def passes(data: dict[str, Any]) -> bool:
        """``data`` has its turn but cannot start yet (it says so in its ticket) and does not
        keep ``ticket`` waiting; see _yields_turn. A GPU entry does not pass another held by
        the GPU: they take the GPU in the lane's order."""
        if mine.get("gpu"):
            return _yields_turn(data, now) and not data.get("blocked_gpu")
        return _yields_turn(data, now) or bool(gpu_taken and data.get("gpu"))

    before = [p for p in order[: order.index(ticket)] if not passes(by_path[p])] \
        if ticket in order else []
    ahead = [by_path[p] for p in before]
    # an entry that gave its slot up while it waits for memory holds nothing; once it has yielded
    # for BLOCKED_YIELD_S it goes before the others again, so a large job is not starved
    ahead = [d for p, d in live if d.get("released") and p != ticket
             and not _yields_turn(d, now)] + ahead
    holders = [d for p, d in live if _holds_lane(d) and p != ticket and not d.get("released")
               and not _yields_turn(d, now)]
    if hold is not None:
        mine = by_path.get(ticket, {})
        if held_for(ticket, mine, hold):
            # before us only the owner's own earlier ones of this hold, and deadlines
            ahead = [by_path[p] for p in before if held_for(p, by_path[p], hold)
                     or by_path[p].get("deadline")]
        elif not mine.get("deadline"):
            left = float(hold["until"]) - time.time()
            return holders, (f"held {max(0.0, left):.0f} s for {hold.get('owner')}, "
                             f"whose {hold.get('who') or 'run'} just ended")
    if not ahead and len(holders) < slots:
        return [], ""
    ahead = holders + ahead
    names = ", ".join(str(d.get("who") or d.get("pid")) for d in ahead[:3])
    return ahead, f"{len(ahead)} ahead: {names}"


LANE_SLOT_DEFAULTS = {DEFAULT_LANE: 2, "light": 2, "codex": 3, "win": 1, "win-cpu": 2, "giz": 2}  # remote lanes: one slot per machine, more for a hosted service


def lane_slots(lane: str) -> int:
    """How many runs of a lane may hold it at once: $STAGELOAD_<LANE>_SLOTS, else 2 in ram, 2 in
    light (29.10: a browser session held its one slot for 100 minutes while builds waited), 3 in
    codex (its jobs call a hosted image service and use no local memory), 1 in win (an 8 GB
    RTX 4060; 30.10: two GPU jobs side by side made Real-CUGAN fail its frames and ended a
    4K master; CPU work goes to win-cpu), 2 in win-cpu (a 20-thread machine whose jobs are often network or disk bound)
    and 1 elsewhere.
    A second ram run still has to fit the budget and pass the start gate, and two --gpu runs never
    run together."""
    try:
        return max(1, int(os.environ[f"STAGELOAD_{lane.upper().replace('-', '_')}_SLOTS"]))
    except (KeyError, ValueError):
        return LANE_SLOT_DEFAULTS.get(lane, 1)


def _slot_locks(lane_dir: Path) -> list[IO[str]]:
    """One lock file per slot: run.lock, then run-1.lock, ... The command inherits the one its run
    took, so the slot stays taken while it runs even if the run is killed with SIGKILL."""
    names = ["run.lock"] + [f"run-{i}.lock" for i in range(1, lane_slots(lane_dir.name))]
    return [(lane_dir / name).open("a") for name in names]


def _take_slot(locks: list[IO[str]]) -> IO[str] | None:
    for lock in locks:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock
        except BlockingIOError:
            continue
    return None


def _nap_until_change(io: GuardIO, seconds: float, signals: _Signals, lane_dir: Path) -> None:
    """Nap in one-second slices until a signal or a change to the lane directory."""
    try:
        mtime = lane_dir.stat().st_mtime_ns
    except OSError:
        return
    while seconds > 0:
        if signals.count:
            return
        duration = min(1.0, seconds)
        io.sleep(duration)
        seconds -= duration
        try:
            if lane_dir.stat().st_mtime_ns != mtime:
                return
        except OSError:
            return


GATE_TTL_S = 30.0
_gate_seen: dict[str, tuple[float, str]] = {}


def lane_gate(lane_dir: Path, ticket: Path | None = None, data: dict[str, Any] | None = None) -> str:
    """A lane's own start check: an executable ``<home>/gates/<lane>`` runs before a run of that
    lane starts and exits non-zero while the lane should take nothing new, for example while the
    machine behind a remote lane is too hot; the first line it prints is the reason the run
    waits. It runs again at most every 30 s. A gate that cannot run does not stop the lane.
    The gate is told which run asks, so it can hold one run (say, a clip another machine is
    already rendering) while the rest of the lane goes: STAGELOAD_GATE_WHO, STAGELOAD_GATE_PID and STAGELOAD_GATE_TICKET;
    each ticket keeps its own answer."""
    gate = lane_dir.parent.parent / "gates" / lane_dir.name
    if not (gate.is_file() and os.access(gate, os.X_OK)):
        return ""
    data = data or {}
    key = f"{gate}|{ticket or ''}"
    now = time.monotonic()
    seen = _gate_seen.get(key)
    if seen and now - seen[0] < GATE_TTL_S:
        return seen[1]
    env = {**os.environ, "STAGELOAD_GATE_WHO": str(data.get("who") or ""),
           "STAGELOAD_GATE_PID": str(data.get("pid") or ""), "STAGELOAD_GATE_TICKET": str(ticket or "")}
    try:
        p = subprocess.run([str(gate)], capture_output=True, text=True, timeout=60, env=env)
        lines = p.stdout.strip().splitlines()
        reason = "" if p.returncode == 0 else (lines[0] if lines else f"the lane's gate says wait (exit {p.returncode})")[:200]
    except (OSError, subprocess.SubprocessError):
        reason = ""
    _gate_seen[key] = (now, reason)
    return reason


def _wait_turn(
    choices: list[tuple[Path, Path, list[IO[str]]]],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    until: float | None = None,
    startable: Callable[[Path, Path], str] | None = None,
    park: Callable[[], None] | None = None,
    gpu: bool = False,
) -> tuple[int, IO[str]] | None:
    """The first of the lanes (lane dir, ticket, slot locks) where the entry has its turn and a
    slot lock is ours, and that lock; None after ``until`` or on a signal. With ``startable``, a
    lane counts only when it also says "" (no reason the run could not start there now); while
    it gives a reason the ticket is marked blocked, so the entries behind may go first."""
    deadline = io.clock() + config.start_timeout if until is None else until
    last = ""
    while True:
        if signals.count:
            return None
        reasons = []
        far_lane = None
        for index, (lane_dir, ticket, locks) in enumerate(choices):
            live = _live_tickets(lane_dir)
            hold = _active_hold(lane_dir)
            ahead, reason = lane_ahead(ticket, live, _last_starts(lane_dir), hold,
                                      lane_slots(lane_dir.name))
            if len(choices) == 1 and reason and len(ahead) > lane_slots(lane_dir.name) + 8:
                far_lane = lane_dir
            mine = dict(live).get(ticket)
            if not reason and startable is not None:
                reason = startable(lane_dir, ticket)
                if reason and mine is not None:
                    _mark_blocked(ticket, mine, reason, time.time(), gpu=gpu)
            if not reason:
                reason = lane_gate(lane_dir, ticket, mine)
                if reason and mine is not None:
                    _mark_blocked(ticket, mine, reason, time.time(), gpu=gpu)
            if not reason:
                if mine is not None:
                    _clear_blocked(ticket, mine)  # the mark is for a turn we could not take
                lock = _take_slot(locks)
                if lock is not None:
                    if held_for(ticket, dict(live).get(ticket, {}), hold):
                        (lane_dir / "hold.json").unlink(missing_ok=True)  # used up
                    return index, lock
                reason = "the lane is still held by a command whose run is gone"
            reasons.append(reason if len(choices) == 1 else f"{lane_dir.name}: {reason}")
        reason = "; ".join(reasons)
        where = ",".join(lane_dir.name for lane_dir, _, _ in choices)
        kind = re.sub(r"\d+(?:\.\d+)?", "#", reason)  # the readings move every poll
        if kind != last:
            io.log(f"queue: waiting in {where} ({reason})")
            last = kind
        if io.clock() >= deadline:
            io.log(f"queue: no turn after {config.start_timeout:.0f} s ({reason})")
            return None
        if far_lane is not None and park is not None:
            park()  # replaces this process; it does not return
        if far_lane is not None:
            # Polling cost grows quadratically with waiters. Far back, many others must start
            # first; nap longer but wake as soon as the lane changes.
            _nap_until_change(io, FAR_POLL_S, signals, far_lane)
        else:
            _nap(io, config.poll, signals)


def _gpu_tickets(home: Path) -> list[tuple[Path, dict[str, Any]]]:
    """GPU entries of every lane that hold their lane's turn: running, or waiting for the GPU."""
    root = home / "queue"
    found = []
    for lane_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        found += [(p, d) for p, d in _live_tickets(lane_dir) if d.get("gpu") and _holds_lane(d)]
    return found


def _gpu_starts(home: Path) -> dict[str, int]:
    try:
        noted = json.loads((home / "queue" / "gpu-owners.json").read_text(encoding="utf-8"))
        return {owner: int(start) for owner, start in noted.items()}
    except (OSError, ValueError, AttributeError, TypeError):
        return {}


def _note_gpu_start(home: Path, owner: str) -> None:
    starts = _gpu_starts(home)
    starts[owner] = time.time_ns()
    _write_json(home / "queue" / "gpu-owners.json", starts)


def gpu_ahead(
    ticket: Path, tickets: list[tuple[Path, dict[str, Any]]], last: dict[str, int]
) -> list[dict[str, Any]]:
    """The GPU entries that go before ``ticket``: one that runs or already holds the GPU, then the
    others at the head of their lanes in fair order, as within a lane."""
    by_path = dict(tickets)
    def has_gpu(d: dict[str, Any]) -> bool:
        return d.get("state") == "running" or bool(d.get("gpu_turn"))

    using = [d for _, d in tickets if has_gpu(d)]
    as_lane = [(p, {**d, "state": "running" if has_gpu(d) else "waiting"}) for p, d in tickets]
    order = fair_order(as_lane, last)
    ahead = [by_path[p] for p in order[: order.index(ticket)]] if ticket in order else []
    return [d for d in using if by_path.get(ticket) is not d] + ahead


def _gpu_reason(ahead: list[dict[str, Any]]) -> str:
    names = ", ".join(f"{d.get('who') or d.get('pid')} ({d.get('lane')})" for d in ahead[:3])
    return f"{len(ahead)} ahead: {names}"


def _wait_gpu(
    ticket: Path,
    home: Path,
    gpu_lock: IO[str],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    until: float,
) -> bool:
    """True once no other GPU job runs in any lane, none waits before ours, and the GPU lock is
    ours.

    A GPU job in the light lane made a render 2.4 times slower (200 s a step instead of 84 s).
    The entry keeps its lane's turn while it waits here.
    """
    last = ""
    while True:
        if signals.count:
            return False
        ahead = gpu_ahead(ticket, _gpu_tickets(home), _gpu_starts(home))
        if not ahead:
            try:
                fcntl.flock(gpu_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                reason = "the GPU is still held by a command whose run is gone"
        else:
            reason = _gpu_reason(ahead)
        if reason != last:
            io.log(f"queue: waiting for the GPU ({reason})")
            last = reason
        if io.clock() >= until:
            io.log(f"queue: no GPU turn after {config.start_timeout:.0f} s ({reason})")
            return False
        _nap(io, config.poll, signals)


def trace_growth(path: Path, now: float | None = None) -> tuple[int, int] | None:
    """How much memory a running job added, by its trace: (since its start, over the last minute).

    Both are measured to the job's peak of the last minute, so a job that is being stopped keeps
    its blame while it frees its memory. None when the trace cannot say: unreadable, no samples,
    or no sample for ``_STALE_S`` seconds.
    """
    stats = trace_stats(path, now)
    return None if stats is None else (stats[0], stats[1])


def trace_stats(path: Path, now: float | None = None) -> tuple[int, int, int] | None:
    """``trace_growth`` and the largest rise of the footprint within the last minute, trough to
    peak (``window_rise``): whether the job grew over the window the guards judge."""
    now = time.time() if now is None else now
    try:
        with path.open("rb") as file:
            head = file.read(8192)
            size = file.seek(0, os.SEEK_END)
            offset = max(0, size - _TRACE_TAIL)
            file.seek(offset)
            tail = file.read()
    except OSError:
        return None

    def records(chunk: bytes, cut: bool) -> list[dict[str, Any]]:
        lines = chunk.split(b"\n")[1 if cut else 0 :]
        out = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # the line cut by the read, or one being written
            if isinstance(record, dict):
                out.append(record)
        return out

    head_records = records(head, False)
    start = next((r for r in head_records if r.get("type") == "start"), None)
    first = next((r for r in head_records if r.get("type") == "sample"), None)
    samples = [r for r in records(tail, offset > 0) if r.get("type") == "sample"]
    if start is None or first is None or not samples:
        return None
    try:
        began = datetime.fromisoformat(start["utc"]).timestamp()
        latest = samples[-1]
        if now - (began + float(latest["t"])) > _STALE_S:
            return None
        window = [r for r in samples if float(r["t"]) >= float(latest["t"]) - RECENT_S]
        values = [int(r["footprint"]) for r in window]
        peak = max(values)
        return (peak - int(first["footprint"]), peak - values[0], window_rise(values))
    except (KeyError, TypeError, ValueError):
        return None


def _ticket_trace(home: Path, data: dict[str, Any], found: dict[int, Path | None]) -> Path | None:
    if data.get("trace"):
        return Path(data["trace"])
    # a run of an older stageload does not name its trace, but its default trace ends with its pid
    pid = int(data.get("pid") or 0)
    if pid not in found:
        names = sorted((home / "traces").glob(f"*-{pid}.jsonl"))
        found[pid] = names[-1] if names else None
    return found[pid]


def _timestamp(value: Any) -> float:
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except (TypeError, ValueError):
        return 0.0


def running_jobs(home: Path, me: Path | None = None) -> Callable[[], list[Growth]]:
    """What a guard reads of the other running jobs: every running entry of every lane whose
    trace can be read, with when it started and whether it is paused. ``me`` is the guard's own
    ticket."""
    found: dict[int, Path | None] = {}

    def jobs() -> list[Growth]:
        out = []
        remote = remote_lanes()
        for lane_dir in sorted(p for p in (home / "queue").iterdir() if p.is_dir()):
            if lane_dir.name in remote:
                continue  # its command runs elsewhere: it neither adds nor frees memory here
            for path, data in _live_tickets(lane_dir):
                if data.get("state") not in _RUNS:
                    continue
                trace = _ticket_trace(home, data, found)
                growth = trace_stats(trace) if trace is not None else None
                if growth is None:
                    continue
                who = str(data.get("who") or data.get("pid"))
                out.append(Growth(who=who, grown=growth[0], recent=growth[1], me=path == me,
                                  rise=growth[2], started=_timestamp(data.get("started")),
                                  paused=data.get("state") == "paused",
                                  gpu=bool(data.get("gpu"))))
        return out

    return jobs


_peaks_cache: dict[str, Any] = {}


def history_peaks(home: Path) -> dict[str, int]:
    """The largest ``peak_footprint`` of each --who in history.jsonl, read again only when the
    file changes."""
    path = home / "history.jsonl"
    try:
        stat = path.stat()
    except OSError:
        return {}
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if _peaks_cache.get("key") == key:
        return _peaks_cache["peaks"]
    peaks: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}
    for line in lines:
        try:
            record = json.loads(line)
            who, peak = str(record.get("who") or ""), int(record.get("peak_footprint") or 0)
        except (ValueError, TypeError, AttributeError):
            continue
        if who and peak > 0:
            peaks[who] = max(peaks.get(who, 0), peak)
    _peaks_cache.update(key=key, peaks=peaks)
    return peaks


def _who_prefix(who: str) -> str:
    """--who without its last dash-separated part: ``owner-task-3`` -> ``owner-task``."""
    return who.rsplit("-", 1)[0] if "-" in who else who


def _numbered(who: str) -> bool:
    """The --who's last part holds a digit (c011, s2, 03): one of a numbered series, whose
    siblings predict it. A numbered step can learn from numbered siblings, but not from a named
    step with the same prefix."""
    return "-" in who and any(ch.isdigit() for ch in who.rsplit("-", 1)[1])


def lane_default_peak(lane: str | None) -> int:
    """The peak of a command nothing else predicts: 16 GB in the ram lane, 4 GB elsewhere."""
    return RAM_LANE_PEAK if (lane or DEFAULT_LANE) == DEFAULT_LANE else DEFAULT_PEAK


def predict_peak(
    cmd: str, who: str, given: int | None, peaks: dict[str, int], lane: str | None = None
) -> tuple[int, str]:
    """A command's predicted peak and where it comes from: --mem-peak; else the larger of the
    engine its command line names and 1.3 times the largest peak the history has for the same
    --who, or, for a numbered --who (its last part holds a digit), the largest of its numbered
    siblings (the same --who without its last -part); else 16 GB in the ram lane and
    4 GB in the others. The history is measured, so it wins over an engine's guess when larger."""
    if given:
        return int(given), "--mem-peak"
    engine = next(((peak, f"command /{pattern.pattern}/") for pattern, peak in PEAK_PATTERNS
                   if pattern.search(cmd)), None)
    seen = peaks.get(who, 0) if who else 0
    if not seen and _numbered(who):
        prefix = _who_prefix(who)
        seen = max((peak for name, peak in peaks.items() if name == prefix or (
            name.startswith(prefix + "-") and _numbered(name)
            and _who_prefix(name) == prefix)), default=0)
    measured = (int(seen * HISTORY_FACTOR), f"history of {who} x{HISTORY_FACTOR}") if seen else None
    found = [x for x in (engine, measured) if x is not None]
    if found:
        return max(found, key=lambda x: x[0])
    if (lane or DEFAULT_LANE) == DEFAULT_LANE:
        return RAM_LANE_PEAK, "default of the ram lane"
    return DEFAULT_PEAK, "default"


def ticket_peak(data: dict[str, Any], peaks: dict[str, int]) -> int:
    """The peak a ticket is counted with: written by its run, or predicted for a run of an older
    stageload that did not write one."""
    try:
        if data.get("mem_peak"):
            return int(data["mem_peak"])
    except (TypeError, ValueError):
        pass
    return predict_peak(str(data.get("cmd") or ""), str(data.get("who") or ""), None, peaks,
                        lane=data.get("lane"))[0]


def budget_reason(
    home: Path, ticket: Path, own: int, config: GuardConfig, ram: int
) -> str:
    """Why the job does not fit the machine's budget ("" when it does): the predicted peaks of the
    running jobs of every lane, private lanes too, plus its own, plus the reserve, over the RAM.
    A job that does not fit even alone starts only when nothing else runs."""
    peaks = history_peaks(home)
    running = []
    root = home / "queue"
    remote = remote_lanes()
    for lane_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in remote):
        for path, data in _live_tickets(lane_dir):
            if path != ticket and data.get("state") in _RUNS:
                running.append((data, ticket_peak(data, peaks)))
    used = sum(peak for _, peak in running)
    if not running or used + own + config.reserve <= ram:
        return ""
    names = ", ".join(f"{d.get('who') or d.get('pid')} ({d.get('lane')}) {peak / GB:.0f} GB"
                      for d, peak in sorted(running, key=lambda r: -r[1])[:3])
    return (f"budget: running {_gb(used)} ({names}) + ours {_gb(own)} + reserve "
            f"{_gb(config.reserve)} > RAM {_gb(ram)}")


def remote_lanes() -> set[str]:
    """Lanes whose jobs run on another machine (STAGELOAD_REMOTE_LANES, comma-separated; default
    win,win-cpu,arch,arch-gpu,modal,codex,giz,gcp: the lane names of the author's setup). These
    jobs use no memory from this machine, so its gates, budget and readings do not hold them."""
    raw = os.environ.get("STAGELOAD_REMOTE_LANES", "win,win-cpu,arch,arch-gpu,modal,codex,giz,gcp")
    return {x.strip() for x in raw.split(",") if x.strip()}


def _ram() -> int:
    try:
        return machine_ram()
    except (OSError, ValueError):
        return 48 * GB


# commands whose memory the history cannot see: they start an app through `open`, and the trace
# reads some 0.01 GB while Godot in film_look.sh holds ~6.9 GB
UNSEEN_BY_HISTORY = re.compile(r"godot|film_look|movie\.sh|flow_check", re.IGNORECASE)


def small_job_limit() -> int:
    """$STAGELOAD_SMALL_JOB_GB (default 1): the largest predicted peak of a small job."""
    try:
        return int(float(os.environ["STAGELOAD_SMALL_JOB_GB"]) * GB)
    except (KeyError, ValueError):
        return GB


def tiny_job_limit() -> int:
    """$STAGELOAD_TINY_JOB_GB (default 0.25): the largest predicted peak of a tiny job."""
    try:
        return int(float(os.environ["STAGELOAD_TINY_JOB_GB"]) * GB)
    except (KeyError, ValueError):
        return int(0.25 * GB)


def is_small_job(entry: dict[str, Any]) -> bool:
    """A job the small-job gate holds instead of the start gate: predicted at most
    $STAGELOAD_SMALL_JOB_GB, not --gpu, on this machine, and not a command the history cannot
    see."""
    try:
        peak = int(entry["mem_peak"])
    except (KeyError, TypeError, ValueError):
        return False
    return (peak <= small_job_limit() and not entry.get("gpu")
            and str(entry.get("lane") or DEFAULT_LANE) not in remote_lanes()
            and not UNSEEN_BY_HISTORY.search(str(entry.get("cmd") or "")))


# 2026-10-28: jobs of 0.25 GB or less waited hours on a swap that only a big render grows; the level floor still holds them
def is_tiny_job(entry: dict[str, Any]) -> bool:
    try:
        peak = int(entry["mem_peak"])
    except (KeyError, TypeError, ValueError):
        return False
    return is_small_job(entry) and peak <= tiny_job_limit()


def small_gate(config: GuardConfig, swap: int, level: int, *, tiny: bool = False) -> list[str]:
    """What holds a small job: swap over gate_stale_compressor (24 GB) or the level under the pause floor.
    30.10: a small job waited on swap 11 GB at 28-31 % available, while the start gate already let a large
    job pass on swap up to 24 GB; swap under that holds idle pages, and a job under 1 GB does not add to it."""
    return over_gate(swap, None, level, swap_max=(1 << 62) if tiny else config.gate_stale_compressor,
                     compressor_max=0,
                     level_min=config.pause_level)


def _local_runs(home: Path, ticket: Path) -> list[dict[str, Any]]:
    """The running and paused entries of this machine's lanes, other than ``ticket``."""
    root, remote, found = home / "queue", remote_lanes(), []
    for lane_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in remote):
        found += [d for p, d in _live_tickets(lane_dir) if p != ticket and d.get("state") in _RUNS]
    return found


def idle_with_stale_swap(
    config: GuardConfig, compressor: int | None, level: int, others: list[dict[str, Any]]
) -> str:
    """Why swap over the gate does not hold the start now, or "" when it still does. macOS does not
    page swapped-out memory back in once memory is free again, so swap can stay above the gate for
    hours. With the compressor at its gate or under, such swap is stale when the memory level is at
    gate_stale_swap_level (35 %) or over (the compressor then up to
    gate_stale_compressor, 24 GB), or when no local queue job runs and the level is at the
    start gate or over."""
    if compressor is None:
        return ""
    if compressor <= config.gate_compressor:
        if not others and level >= config.gate_level:
            return (
                f"no local queue job runs, memory available is {level}% and the compressor "
                f"{_gb(compressor)}"
            )
        if level >= config.gate_stale_swap_level:
            return (
                f"memory available is {level}% (>= {config.gate_stale_swap_level}%) and the "
                f"compressor {_gb(compressor)}"
            )
    elif level >= config.gate_stale_swap_level and compressor <= config.gate_stale_compressor:
        return (
            f"memory available is {level}% (>= {config.gate_stale_swap_level}%) and the compressor "
            f"{_gb(compressor)} holds idle pages (<= {_gb(config.gate_stale_compressor)})"
        )
    return ""


def _gate_for(
    entry: dict[str, Any], config: GuardConfig, io: GuardIO, home: Path, ticket: Path
) -> tuple[list[str], str]:
    """The gate's reasons to wait, and a note when swap over it was let pass on an idle Mac."""
    swap, level = io.swap_used(), io.level()
    compressor = read_optional(io.compressor)
    if is_small_job(entry):
        reasons = small_gate(config, swap, level, tiny=is_tiny_job(entry))
    else:
        reasons = start_gate(config, swap, compressor, level)
    note = ""
    # swap, and the compressor up to gate_stale_compressor, hold only pages of idle processes once
    # enough memory is available (29.10: 38 % available, compressor 18 GB, nothing started)
    stale = [r for r in reasons if r.startswith("swap ") or (
        r.startswith("compressor ") and compressor is not None
        and compressor <= config.gate_stale_compressor and level >= config.gate_stale_swap_level)]
    why = (
        idle_with_stale_swap(config, compressor, level, _local_runs(home, ticket)) if stale else ""
    )
    if why:
        reasons = [r for r in reasons if r not in stale]
        note = f"{stale[0]}, but {why}: the swap is stale"
    disk = disk_reason(home)
    if disk:
        reasons = reasons + [disk]
    return reasons, note


PLACE_HOLD_S = 600.0


def _place_file(home: Path, who: str) -> Path:
    return home / "places" / (re.sub(r"[^A-Za-z0-9._-]+", "_", who) + ".json")


def keep_place(home: Path, who: str, ns: str) -> None:
    """`stageload reload --end-for-retry` ended a waiting run whose script queues the same --who
    again: its ticket's time is kept here for PLACE_HOLD_S, so the retry gets the same place."""
    path = _place_file(home, who)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(path, {"ns": ns, "until": time.time() + PLACE_HOLD_S})


def take_place(home: Path, who: str) -> str:
    """The ticket time kept for ``who``, once, or "" when none is kept or it ran out."""
    if not who:
        return ""
    path = _place_file(home, who)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        path.unlink()
    except (OSError, ValueError):
        return ""
    ns = str(data.get("ns") or "")
    return ns if ns.isdigit() and float(data.get("until") or 0) > time.time() else ""


def disk_free(home: Path) -> int:
    """Free bytes on the volume of the queue home (os.statvfs), what a new job's output lands on
    here."""
    st = os.statvfs(home)
    return st.f_bavail * st.f_frsize


def disk_reason(home: Path) -> str:
    """No new job starts on this machine while its data volume has under $STAGELOAD_GATE_DISK_GB
    free (8 GiB by default: only an emergency floor, so a full disk cannot stop the system). Running jobs are
    never touched."""
    try:
        limit = float(os.environ.get("STAGELOAD_GATE_DISK_GB", "8")) * GB
        free = disk_free(home)
    except (OSError, ValueError):
        return ""
    return f"disk: {free / GB:.1f} GiB free < {limit / GB:.0f} GiB" if free < limit else ""


def _gate_words(entry: dict[str, Any], config: GuardConfig) -> str:
    if is_tiny_job(entry):
        return (f"tiny-job gate: available {config.pause_level}% only "
                f"(peak at most {_gb(tiny_job_limit())})")
    if is_small_job(entry):
        return (f"small-job gate swap {_gb(config.gate_stale_compressor)}, available {config.pause_level}% "
                f"(peak at most {_gb(small_job_limit())})")
    return f"start gate {gate_text(config)}"


def _wait_memory(
    ticket: Path,
    home: Path,
    entry: dict[str, Any],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    until: float,
    slot: IO[str] | None = None,
    locks: Sequence[IO[str]] = (),
) -> tuple[bool, IO[str] | None]:
    """(True, the slot lock held) once the machine's readings are under the start gate, the
    guard's own room checks (--wait-free, --swap-limit, --busy and --quiet-for) pass and the job
    fits the budget; the ticket is then marked running while the budget's lock is held, so a
    guard of another lane that checks next counts it. Meanwhile the ticket holds its lane in the
    state ``mem-wait`` and says why.

    With ``slot``, the slot is given up while the job waits (its ticket says ``released``, and it
    is no holder of the lane), and taken back, any free one of ``locks``, when the way is clear:
    without blocking, so a signal still ends the wait and the lane never shows two holders for one
    running job. Without ``slot`` (a --gpu job, which holds the GPU's turn) the slot is kept.

    A render, a 12-worker CPU job on another lane, a build and Blender jobs ran at once; the
    compressor reached 15 GB and only 0.1 GB was free. The render's footprint missed Metal memory,
    and each lane waited only for itself.
    """
    ram = _ram()
    own = int(entry.get("mem_peak") or lane_default_peak(entry.get("lane")))
    last: tuple[str, ...] = ()
    released = False
    regexes = [re.compile(pattern) for pattern in config.busy]
    quiet: dict[str, float | None] = {"since": None}

    def idle() -> bool:
        return bool(idle_with_stale_swap(config, read_optional(io.compressor), io.level(),
                                         _local_runs(home, ticket)))

    with (home / "queue" / "budget.lock").open("a") as budget_lock:
        while True:
            if signals.count:
                return False, slot
            reasons, note = _gate_for(entry, config, io, home, ticket)
            # 30.10: --wait-free (default 35 %) held small jobs at 27 % although their own gate
            # floor is the pause level (25 %); for a small job only its own gate counts
            small = is_small_job(entry)
            reasons += [r for r in _no_room_reasons(config, io, regexes, quiet, idle)
                        if r not in reasons and not (small and r.startswith("memory available"))]  # --wait-free at the gate's level says it once
            if released and not reasons:
                # the way is clear: take a slot back, after whoever ran in it meanwhile. The
                # blocked marks stay until the real start: a job that waited long keeps its claim
                taken = _take_slot(list(locks) or [slot])
                if taken is None:
                    reasons.append("the lane's slots are taken by runs that passed while we waited")
                else:
                    slot, released = taken, False
                    entry.pop("released", None)
                    _write_json(ticket, entry)
                    continue
            if not reasons:
                fcntl.flock(budget_lock.fileno(), fcntl.LOCK_EX)
                try:
                    over = budget_reason(home, ticket, own, config, ram)
                    if over:
                        reasons.append(over)
                    if not reasons:
                        entry.pop("mem_wait", None)
                        for key in _BLOCKED_KEYS:
                            entry.pop(key, None)
                        entry.update(state="running", started=_utc())
                        _write_json(ticket, entry)
                        if note:
                            io.log(f"queue: starting although {note}")
                        if own + config.reserve > ram:
                            io.log(f"queue: predicted peak {_gb(own)} + reserve "
                                   f"{_gb(config.reserve)} > RAM {_gb(ram)}: it starts alone")
                        return True, slot
                finally:
                    fcntl.flock(budget_lock.fileno(), fcntl.LOCK_UN)
            reason = "; ".join(reasons)
            kinds = tuple(r.split(" ", 1)[0] for r in reasons)
            now = time.time()
            if slot is not None and not released:
                # 29.10: a run that waited here kept its lane's only slot, and three runs that
                # could have started waited behind it. The slot is free while we wait.
                fcntl.flock(slot.fileno(), fcntl.LOCK_UN)
                released = True
            stale = now - float(entry.get("blocked_at") or 0) >= BLOCKED_REFRESH_S
            if entry.get("mem_wait") != reason or (released and stale):
                entry.update(state="mem-wait", mem_wait=reason)
                entry.setdefault("turn", _utc())
                if released:
                    entry.update(blocked=reason, blocked_at=now, released=True)
                    entry.setdefault("blocked_since", now)
                _write_json(ticket, entry)
            if kinds != last:
                io.log(f"queue: waiting for memory ({reason}; {_gate_words(entry, config)}, "
                       f"our predicted peak {_gb(own)} by {entry.get('mem_peak_from')})")
                last = kinds
            if io.clock() >= until:
                io.log(f"queue: no memory after {config.start_timeout:.0f} s ({reason})")
                return False, slot
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
    owner: str | None = None,
    deadline: str | None = None,
    gpu: bool = False,
    mem_peak: int | None = None,
    what: str = "",
    cwd: str | None = None,
    run_args: Sequence[str] | None = None,
    since: str | None = None,
) -> int:
    """Run ``cmd`` in its turn. ``lane`` may list several lanes, comma-separated («win,light»):
    the entry waits in each, and the first where it can start now (its turn, a free slot, and for
    a lane of this machine the start gate, the budget and the GPU) takes it; the entries in the
    others are dropped. The command gets the lane in $STAGELOAD_LANE.

    The ticket and the history keep ``cmd`` as ``argv``, ``cwd`` and ``run_args`` (the flags of
    `stageload run` itself), so `stageload requeue` can restore the command after a signal ends
    it. ``since`` is when the run was first queued (a parked run started again): its ticket shows
    it, and the start timeout counts from it."""
    parkable = io is None and os.environ.get("STAGELOAD_PARK", "1") != "0"
    io = io or system_io()
    if cwd is None:
        try:
            cwd = os.getcwd()
        except OSError:
            cwd = None
    home = Path(home) if home is not None else default_home()
    lanes = list(dict.fromkeys(x.strip() for x in lane.split(",") if x.strip())) or [DEFAULT_LANE]
    io = replace(io, processes=_without_wrappers(io.processes, home))
    # `stageload reload` hands a run the time of the ticket it replaces, so the new run keeps
    # that place in the lane;
    # popped, so a `stageload run` inside the command takes a time of its own
    keep_ns, _, keep_who = os.environ.pop("STAGELOAD_TICKET_NS", "").partition(":")
    keep = keep_ns.isdigit() and keep_who == who  # «ns:who»: a wrapper's other runs
    # (h3_shot's -up) take their own time
    if not keep:
        keep_ns = take_place(home, who)
        keep = bool(keep_ns)
    name = f"{int(keep_ns) if keep else time.time_ns():020d}-{os.getpid()}.json"
    peaks = history_peaks(home)
    choices: list[tuple[Path, Path, list[IO[str]]]] = []
    entries: list[dict[str, Any]] = []
    for one in lanes:
        lane_dir = home / "queue" / one
        lane_dir.mkdir(parents=True, exist_ok=True)
        entry = {"pid": os.getpid(), "who": who, "owner": owner_of(who, owner), "lane": one,
                 "cmd": " ".join(cmd)[:300], "since": since or _utc(), "state": "waiting",
                 "argv": list(cmd), "cwd": cwd,
                 "run_args": list(run_args) if run_args is not None else None}
        if len(lanes) > 1:
            entry["lanes"] = lanes
        if what:
            entry["what"] = what  # what it makes and what for, in plain English
        if deadline:
            entry["deadline"] = deadline
        if gpu and one not in remote_lanes():
            # a GPU on another machine is not this machine's GPU: a remote lane never takes the local GPU turn
            entry["gpu"] = True
        # what the other guards sum to see whether a job fits the machine
        entry["mem_peak"], entry["mem_peak_from"] = predict_peak(entry["cmd"], who, mem_peak,
                                                                 peaks, lane=one)
        _write_json(lane_dir / name, entry)
        entries.append(entry)
        choices.append((lane_dir, lane_dir / name, _slot_locks(lane_dir)))
    _write_queued(home, entries[0])
    queued_at = io.clock()
    if since:
        # a parked run started again: the 24 h to start count from when it was first queued
        queued_at -= max(0.0, time.time() - _timestamp(since))
    until = queued_at + config.start_timeout
    lock: IO[str] | None = None
    gpu_lock = (home / "queue" / "gpu.lock").open("a") if gpu else None
    room = False
    ticket = choices[0][1]

    def startable(lane_dir: Path, ticket: Path) -> str:
        """Why the run could not start in this lane now ("" when it could): for a run of several
        lanes the gate and the budget, for a --gpu run the GPU. A --gpu run takes no slot while
        another --gpu run holds the GPU (29.10: one in gpu-wait held the second ram slot for the
        whole render, and the jobs without --gpu behind it waited for nothing)."""
        if lane_dir.name in remote_lanes():
            return ""
        mine = entries[[c[1] for c in choices].index(ticket)]
        reasons = []
        if len(choices) > 1:
            reasons, _ = _gate_for(mine, config, io, home, ticket)
            over = budget_reason(home, ticket, int(mine["mem_peak"]), config, _ram())
            if over:
                reasons.append(over)
        if gpu:
            ahead = gpu_ahead(ticket, _gpu_tickets(home), _gpu_starts(home))
            if ahead:
                reasons.append(f"the GPU is taken, {_gpu_reason(ahead)}")
        return "; ".join(reasons)

    try:
        with _Signals() as signals:
            def park() -> None:
                _park(choices[0][0], ticket, cmd, config, home=home, lane=lanes[0], who=who,
                      trace=trace, owner=owner, deadline=deadline, gpu=gpu, mem_peak=mem_peak,
                      what=what, cwd=cwd, run_args=run_args)

            taken = _wait_turn(choices, config, io, signals, until,
                               startable if len(choices) > 1 or gpu else None,
                               park if parkable and len(choices) == 1 else None, gpu=gpu)
            turn = taken is not None
            if taken is not None:
                index, lock = taken
                lane_dir, ticket, _ = choices[index]
                entry = entries[index]
                for other, (_, path, locks) in enumerate(choices):
                    if other != index:
                        # the other slot locks of our lane stay open: _wait_memory may give the
                        # slot up and take any free one back
                        path.unlink(missing_ok=True)
                        for unused in locks:
                            unused.close()
                if len(choices) > 1:
                    io.log(f"queue: took the {lane_dir.name} lane of {','.join(lanes)}")
            else:
                lane_dir, entry = choices[0][0], entries[0]
            if lane_dir.name == DEFAULT_LANE:
                # the shared list names the engines that fill the memory; a lane such as codex
                # waits on a remote quota instead, and holding it behind those engines only
                # wastes its turn
                config = replace(config, busy=tuple(config.busy) + load_busy(home))
            if turn and gpu_lock is not None:
                entry.update(state="gpu-wait", turn=_utc())
                _write_json(ticket, entry)
                turn = _wait_gpu(ticket, home, gpu_lock, config, io, signals, until)
                if turn:
                    entry["gpu_turn"] = True
                    _write_json(ticket, entry)
            # A lane that runs its command on another machine over ssh uses none of this Mac's
            # memory, so the local start gate, the room checks and the budget do not hold it.
            if turn and lane_dir.name in remote_lanes():
                room = True
                # nor do its readings pause or stop it once it runs; it is still traced
                config = replace(config, watch_memory=False)
            elif turn:
                # a --gpu run already holds the GPU turn here: it keeps its slot, so the next
                # --gpu run cannot take the slot and wait for the GPU this one holds (29.10)
                room, retaken = _wait_memory(ticket, home, entry, config, io, signals, until,
                                             None if gpu else lock, choices[index][2])
                lock = retaken or lock
            if gpu:
                # never paused: stopped at once at the kill floor instead
                config = replace(config, pausable=False,
                                 kill_free=max(config.kill_free, GPU_KILL_LEVEL))
            if signals.first is not None:
                io.log(f"queue: got {_signal_name(signals.first)} before starting the command")
                # a line in the history even though nothing ran: `stageload requeue` finds it there
                _write_history(home, {**_record_of(entry), "exit": 128 + signals.first,
                                      "state": "waiting",
                                      "waited_s": round(io.clock() - queued_at, 1), "run_s": 0.0,
                                      "peak_footprint": None, "paused_s": 0.0, "trace": None})
                return 128 + signals.first
            if not turn or not room:
                return EXIT_NO_ROOM
            entry.update(state="running", started=_utc())
            if trace is not None:
                entry["trace"] = str(trace)  # the other guards read it to tell whose swap it is
            _write_json(ticket, entry)
            _note_start(lane_dir, entry["owner"])
            if gpu_lock is not None:
                _note_gpu_start(home, entry["owner"])
            waited = io.clock() - queued_at
            io = replace(io, jobs=running_jobs(home, ticket))
            if io.neighbours is top_resident:
                # outside the queue: leave out every queued run and what it started, not only ours
                io = replace(io, neighbours=lambda: top_resident(_queue_pids(home)))
            if io.outside is outside_footprint:
                io = replace(io, outside=lambda: outside_footprint(_queue_pids(home)))
            assert lock is not None
            locks = (lock,) if gpu_lock is None else (lock, gpu_lock)
            began = time.monotonic()
            code = _run(cmd, config, io, signals, locks, home, entry, trace, waited, ticket)
            hold = _hold_seconds(_quick_failure(code, time.monotonic() - began))
            if hold > 0:
                _hold_lane(lane_dir, entry["owner"], who, hold)
            return code
    finally:
        for _, path, locks in choices:
            path.unlink(missing_ok=True)
            for one_lock in locks:
                one_lock.close()
        if gpu_lock is not None:
            gpu_lock.close()


def _queue_pids(home: Path) -> set[int]:
    pids = {os.getpid()}
    root = home / "queue"
    for lane_dir in (p for p in root.iterdir() if p.is_dir()):
        pids.update(int(d["pid"]) for _, d in _live_tickets(lane_dir) if d.get("pid"))
    return pids


def _state_writer(
    ticket: Path | None, entry: dict[str, Any], pauses: list[float]
) -> Callable[[str], None]:
    """What the guard calls when it pauses or resumes the command: the ticket shows the state,
    and ``pauses`` keeps the seconds paused (an open pause as a negative start time)."""

    def note(state: str) -> None:
        if state == "paused":
            entry.update(state="paused", paused_since=_utc())
            pauses.append(-time.monotonic())
        else:
            entry["state"] = "running"
            entry.pop("paused_since", None)
            if pauses and pauses[-1] < 0:
                pauses[-1] = time.monotonic() + pauses[-1]
        if ticket is not None:
            _write_json(ticket, entry)

    return note


def _paused_seconds(pauses: list[float]) -> float:
    now = time.monotonic()
    return sum(now + p if p < 0 else p for p in pauses)


def _run(
    cmd: Sequence[str],
    config: GuardConfig,
    io: GuardIO,
    signals: _Signals,
    locks: tuple[IO[str], ...],
    home: Path,
    entry: dict[str, Any],
    trace: Path | None,
    waited: float,
    ticket: Path | None = None,
) -> int:
    swap_at_start = io.swap_used()
    limit = swap_at_start + config.swap_budget
    if config.swap_limit is not None and swap_at_start >= config.swap_limit:
        # If a run starts above its --swap-limit, set a fresh limit from the starting swap plus its
        # budget; otherwise the first sample would stop it before it can run.
        io.log(
            f"queue: swap {_gb(swap_at_start)} is over this run's limit "
            f"{_gb(config.swap_limit)} at the start: "
            f"its limit is {_gb(limit)}, the start plus its budget"
        )
        config = replace(config, swap_limit=limit)
    if config.swap_limit is not None:
        limit = min(limit, config.swap_limit)
    watch = (f"stop above {_gb(limit)} or under {config.kill_free}% available twice"
             if config.watch_memory else "it runs on another machine: never paused or stopped "
             "for this machine's memory")
    io.log(f"queue: start in {entry['lane']} after {waited:.0f} s, swap {_gb(swap_at_start)}, "
           f"{watch}")
    try:
        # the command keeps the lane's lock (and the GPU's) open: they stay taken while it runs
        # the lane it runs in: a run of several lanes picks a remote or a local tool by it
        env = {**os.environ, "STAGELOAD_LANE": entry["lane"]}
        proc = subprocess.Popen(list(cmd), start_new_session=True, env=env,
                                pass_fds=tuple(lock.fileno() for lock in locks))
    except OSError as error:
        io.log(f"queue: cannot start {cmd[0]!r}: {error.strerror or error}")
        return EXIT_CANNOT_START
    started = io.clock()
    entry["pgid"] = proc.pid  # what the queue resumes if this run dies while the command is paused
    if ticket is not None:
        _write_json(ticket, entry)
    pauses: list[float] = []
    io = replace(io, on_state=_state_writer(ticket, entry, pauses))
    meter = None
    if trace is not None:
        readers = Readers(footprint=lambda: tree_footprint(proc.pid), swap_used=io.swap_used,
                          level=io.level, compressor=io.compressor, wired=io.wired)
        meta = {"who": entry["who"], "lane": entry["lane"], "cmd": entry["cmd"]}
        meter = MemoryMeter(trace, meta=meta, readers=readers).__enter__()
    try:
        code = _watch(proc, config, io, signals, limit, swap_at_start)
    except BaseException:
        _stop_group(proc, config.grace, signals)
        raise
    finally:
        if meter is not None:
            meter.__exit__(None, None, None)
    summary = meter.summary() if meter is not None else {}
    swap = summary.get("swap", {})
    record = {
        **_record_of(entry),
        "exit": code,
        "waited_s": round(waited, 1),
        "run_s": round(io.clock() - started, 1),
        "peak_footprint": summary.get("peak_footprint"),
        "swap_start": swap.get("start"),
        "swap_peak": swap.get("peak"),
        "swap_end": swap.get("end"),
        "min_level": summary.get("min_level"),
        "paused_s": round(_paused_seconds(pauses), 1),
        "trace": str(trace) if trace is not None else None,
    }
    _write_history(home, record)
    peak = record["peak_footprint"]
    io.log(
        f"queue: exit {code} after {record['run_s']:.0f} s"
        + (f" (paused {record['paused_s']:.0f} s of them)" if record["paused_s"] else "")
        + (f", peak {_gb(peak)}" if peak else "")
        + (f", trace {trace}" if trace is not None else "")
    )
    return code


def _record_of(entry: dict[str, Any]) -> dict[str, Any]:
    """The fields of a history line that come from the ticket: who and what ran, and how to run it
    again (``argv``, ``cwd``, ``run_args``; None for a run of an older stageload)."""
    return {"end": _utc(), "lane": entry["lane"], "who": entry["who"],
            "what": entry.get("what") or "", "cmd": entry["cmd"], "pid": entry.get("pid"),
            "owner": entry.get("owner"), "argv": entry.get("argv"), "cwd": entry.get("cwd"),
            "run_args": entry.get("run_args")}


def _write_history(home: Path, record: dict[str, Any]) -> None:
    with (home / "history.jsonl").open("a", encoding="utf-8") as history:
        history.write(json.dumps(record) + "\n")


QUEUED_KEEP = 2 * 86400  # a waiting run older than this is not sent again by `requeue`


def _write_queued(home: Path, entry: dict[str, Any]) -> None:
    """One line per waiting run in queued.jsonl, written when it joins the queue. 2026-11-04 a
    restart of the Mac ended about 85 waiting runs at once and most of them died before they
    could write their history line, so `requeue` found nothing; this line lets it find them."""
    record = {k: v for k, v in _record_of(entry).items() if k != "end"}
    record["queued"] = _utc()
    path = home / "queued.jsonl"
    try:
        if path.stat().st_size > 4_000_000:  # keep the file small: the last half is plenty
            lines = path.read_text(encoding="utf-8").splitlines()
            path.write_text("".join(x + "\n" for x in lines[len(lines) // 2:]), encoding="utf-8")
    except OSError:
        pass
    with path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(record) + "\n")


def write_spec(
    cmd: Sequence[str],
    config: GuardConfig,
    *,
    home: Path,
    lane: str = DEFAULT_LANE,
    who: str = "",
    trace: Path | None = None,
    owner: str | None = None,
    deadline: str | None = None,
    gpu: bool = False,
    mem_peak: int | None = None,
    what: str = "",
    cwd: str | None = None,
    run_args: Sequence[str] | None = None,
    since: str | None = None,
) -> Path:
    """The run written to a file, for a process that reads it instead of its command line."""
    jobs = home / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    spec = jobs / f"{time.time_ns():020d}-{os.getpid()}.json"
    data = {"cmd": list(cmd), "config": asdict(config), "home": str(home), "lane": lane,
            "who": who, "owner": owner, "deadline": deadline, "gpu": gpu, "mem_peak": mem_peak,
            "what": what, "cwd": cwd,
            "run_args": list(run_args) if run_args is not None else None,
            "trace": str(trace) if trace is not None else None, "since": since}
    _write_json(spec, data)
    return spec


def run_spec(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    path.unlink(missing_ok=True)
    fields = data["config"]
    fields["busy"] = tuple(fields.get("busy", ()))
    trace = Path(data["trace"]) if data.get("trace") else None
    return run_queued(data["cmd"], GuardConfig(**fields), home=Path(data["home"]),
                      lane=data["lane"], who=data["who"], trace=trace, owner=data.get("owner"),
                      deadline=data.get("deadline"), gpu=bool(data.get("gpu")),
                      mem_peak=data.get("mem_peak"), what=data.get("what") or "",
                      cwd=data.get("cwd"), run_args=data.get("run_args"),
                      since=data.get("since"))


# --- cancel and requeue ---------------------------------------------------------------------
# A signal aimed at queue processes can end unrelated jobs too. `cancel` targets named tickets;
# `requeue` sends again the runs a signal ended.

KILLED_EXITS = (143, 137, 130, -15)  # SIGTERM, SIGKILL, SIGINT as 128+n, and SIGTERM as Popen
# gives it


def _all_tickets(home: Path) -> list[tuple[Path, dict[str, Any]]]:
    root = home / "queue"
    if not root.exists():
        return []
    found = []
    for lane_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        found += _live_tickets(lane_dir)
    return found


def matching_tickets(home: Path, who: str | None = None, owner: str | None = None,
                     lane: str | None = None) -> list[dict[str, Any]]:
    """The live tickets that ``who`` (a glob on the ticket's who) or ``owner`` (exact, as
    owner_of() gives it) names, in ``lane`` if given; one row per run, so a run waiting in
    several lanes is ended once."""
    rows: dict[int, dict[str, Any]] = {}
    for path, data in _all_tickets(home):
        if lane is not None and path.parent.name != lane:
            continue
        name = str(data.get("who") or "")
        if who is not None and not fnmatch.fnmatchcase(name, who):
            continue
        if owner is not None and _owner(data) != owner:
            continue
        pid = int(data.get("pid") or path.stem.split("-")[1])
        row = rows.get(pid)
        if row is None:
            rows[pid] = {**data, "pid": pid, "lane": path.parent.name,
                         "state": data.get("state", "waiting"), "owner": _owner(data)}
        else:
            row["lane"] += f",{path.parent.name}"
            if _holds_lane(data):
                row["state"] = data.get("state")
    return list(rows.values())


def _ended(pid: int) -> bool:
    """Gone, or a zombie that only waits for its parent to reap it."""
    if not _alive(pid):
        return True
    try:
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True,
                              text=True, check=False).stdout.strip()
    except OSError:
        return False
    return not stat or stat.startswith("Z")


def stop_pgid(pgid: int, grace: float) -> None:
    """The group of a running command stopped as the guard stops it (guard._stop_group): SIGTERM
    to the group, SIGCONT for a paused one, then SIGKILL to what is left after ``grace`` s."""
    try:
        if _signal_group(pgid, signal.SIGTERM):
            _signal_group(pgid, signal.SIGCONT)
            deadline = time.monotonic() + grace
            while time.monotonic() < deadline and _group_alive(pgid):
                time.sleep(0.05)
            _signal_group(pgid, signal.SIGKILL)
    except Exception:  # noqa: BLE001 - a group we cannot signal is not ours to stop
        pass


def asker() -> dict[str, str]:
    """Who asked for a cancel: the login, and $STAGELOAD_WHO when set."""
    try:
        login = os.getlogin()
    except OSError:
        import getpass
        login = getpass.getuser()
    out = {"asker": login}
    if os.environ.get("STAGELOAD_WHO"):
        out["asker_who"] = os.environ["STAGELOAD_WHO"]
    return out


def cancel(home: Path, rows: list[dict[str, Any]], grace: float = 5.0,
           log: Callable[[str], None] = print) -> list[dict[str, Any]]:
    """End each run of ``rows`` (from matching_tickets). A waiting run gets SIGTERM: its _Signals
    turn that into an exit before its command starts. A running one gets SIGTERM too (its guard
    stops its group and writes its history line), and its group is stopped here as well, in case
    the run cannot. One line per run goes to cancelled.jsonl."""
    try:
        own_group = os.getpgid(0)
    except OSError:
        own_group = -1
    who_asked = asker()
    ended = []
    for row in rows:
        pid, pgid = int(row["pid"]), row.get("pgid")
        if pgid and int(pgid) == own_group:
            log(f"skip   {row['lane']:6} {row.get('who') or '-':20} pid {pid}: "
                f"this cancel runs inside that command")
            continue
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
        except PermissionError as error:
            log(f"skip   {row['lane']:6} {row.get('who') or '-':20} pid {pid}: {error}")
            continue
        if row["state"] in _RUNS and pgid:
            stop_pgid(int(pgid), grace)
        deadline = time.monotonic() + grace + 2.0
        while time.monotonic() < deadline and not _ended(pid):
            time.sleep(0.05)
        line = {"utc": _utc(), "who": row.get("who") or "", "owner": row["owner"],
                "lane": row["lane"], "state": row["state"], "pid": pid,
                "what": row.get("what") or "", **who_asked}
        with (home / "cancelled.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")
        log(f"ended  {row['lane']:6} {row['state']:8} {row.get('who') or '-':20} pid {pid:<7} "
            f"{'' if _ended(pid) else '(still ending) '}{str(row.get('cmd') or '')[:60]}")
        ended.append(line)
    return ended


def _since(text: str) -> float:
    value = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.timestamp()


def requeue_plan(home: Path, since: float, owner: str | None = None
                 ) -> list[tuple[dict[str, Any], str]]:
    """The history records to send again, each with "" or why it cannot be: the latest of each
    who that a signal ended (exit 143, 137, 130 or -15) after ``since``, whose who has no live
    ticket now and no successful run after it, and that `stageload cancel` did not end."""
    try:
        lines = (home / "history.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    cancelled = set()
    try:
        for line in (home / "cancelled.jsonl").read_text(encoding="utf-8").splitlines():
            try:
                c = json.loads(line)
                cancelled.add((c.get("who"), c.get("pid")))
            except (ValueError, AttributeError):
                continue
    except OSError:
        pass
    live = {str(d.get("who") or "") for _, d in _all_tickets(home)}
    done: dict[str, float] = {}
    killed: dict[str, tuple[float, dict[str, Any]]] = {}
    for line in lines:
        try:
            record = json.loads(line)
            end = _timestamp(record.get("end"))
            code = int(record.get("exit"))
        except (ValueError, TypeError, AttributeError):
            continue
        key = str(record.get("who") or "") or str(record.get("cmd") or "")
        if code == 0:
            done[key] = max(done.get(key, 0.0), end)
        elif code in KILLED_EXITS and end > since and end >= killed.get(key, (0.0, {}))[0]:
            killed[key] = (end, record)
    # runs that joined the queue and never wrote a history line: killed too fast to write one
    ended = set()
    for line in lines:
        try:
            r = json.loads(line)
            ended.add((str(r.get("who") or ""), r.get("pid")))
        except (ValueError, AttributeError):
            continue
    try:
        queued_lines = (home / "queued.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError:
        queued_lines = []
    for line in queued_lines:
        try:
            record = json.loads(line)
            at = _timestamp(record.get("queued"))
        except (ValueError, TypeError, AttributeError):
            continue
        key = str(record.get("who") or "") or str(record.get("cmd") or "")
        if at < since - QUEUED_KEEP or (str(record.get("who") or ""), record.get("pid")) in ended:
            continue
        if at >= killed.get(key, (0.0, {}))[0]:
            killed[key] = (at, {**record, "end": record.get("queued"), "exit": "lost"})
    plan = []
    for key, (end, record) in sorted(killed.items(), key=lambda kv: kv[1][0]):
        who = str(record.get("who") or "")
        if owner is not None and (record.get("owner") or owner_of(who)) != owner:
            continue
        if who and who in live:
            continue
        if done.get(key, 0.0) > end or (who, record.get("pid")) in cancelled:
            continue
        if record.get("run_args") is None or not record.get("argv"):
            plan.append((record, "no run_args (older than this version)"))
        elif not record.get("cwd") or not Path(record["cwd"]).is_dir():
            plan.append((record, f"its cwd {record.get('cwd')!r} is gone"))
        else:
            plan.append((record, ""))
    return plan


# set explicitly in run_args by `stageload run`: the requeuer's own values must not leak in
_RUN_ENV = ("STAGELOAD_WHO", "STAGELOAD_WHAT", "STAGELOAD_OWNER", "STAGELOAD_DEADLINE",
            "STAGELOAD_GPU")


def requeue(home: Path, plan: list[tuple[dict[str, Any], str]], dry: bool = False,
            log: Callable[[str], None] = print) -> list[subprocess.Popen[bytes]]:
    """Start each record of ``plan`` again, detached, in its cwd; its output goes to
    requeued/<who>-<time>.log. ``dry`` only prints."""
    started = []
    env = {k: v for k, v in os.environ.items() if k not in _RUN_ENV}
    env["STAGELOAD_HOME"] = str(home)
    for record, why in plan:
        who = str(record.get("who") or "-")
        head = f"{record.get('lane') or '-':6} exit {record.get('exit')!s:<4} {who:20}"
        if why:
            log(f"cannot requeue: {why}  {head}  {str(record.get('cmd') or '')[:60]}")
            continue
        argv = [sys.executable, "-m", "stageload", "run", *record["run_args"], "--",
                *record["argv"]]
        if dry:
            log(f"would requeue  {head}  in {record['cwd']}: {' '.join(record['argv'])[:80]}")
            continue
        logs = home / "requeued"
        logs.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", who) or "job"
        path = logs / f"{name}-{stamp}.log"
        with path.open("ab") as out:
            proc = subprocess.Popen(argv, cwd=record["cwd"], env=env, start_new_session=True,
                                    stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        log(f"requeued       {head}  pid {proc.pid}, log {path}")
        started.append(proc)
    return started


# --- parking --------------------------------------------------------------------------------
# 29.10: about 90 waiting runs each kept a Python process (~25 MB, most of it swapped or
# compressed), and macOS frees such pages only when the process ends. A run far back in its lane
# now replaces itself with a small shell loop under the same pid, so its ticket and its place
# stay; once a minute a short check asks whether it is near the head, and then the loop starts
# the waiter again with the same place (STAGELOAD_TICKET_NS), the deadline its ticket has by then
# (the cleaner may have set one) and the time it was first queued. A signal ends the loop as it
# ended the waiter: a history line with the state «waiting», so `stageload requeue` finds it.

PARK_CHECK_S = 60
NEAR_EXTRA = 4  # resume when at most slots + NEAR_EXTRA runs are ahead
NEAR_FAR = 3  # the exit code of --near for «not yet»; any other failure counts towards NEAR_FAILS
NEAR_FAILS = 3  # checks that failed in a row before the loop gives up checking and resumes
PARK_MARK = "stageload-park-loop"  # in the park loop's text, so a guard pausing a group can leave the loop out
_PARK_LOOP = f"""
: {PARK_MARK}
ended() {{ trap - TERM INT HUP; "$0" -m stageload.queue --ended "$2" "$4" "$5" >/dev/null 2>&1; exit "$5"; }}
trap 'ended "$1" "$2" "$3" "$4" 143' TERM
trap 'ended "$1" "$2" "$3" "$4" 130' INT
trap 'ended "$1" "$2" "$3" "$4" 129' HUP
fails=0
while :; do
  sleep {PARK_CHECK_S} & wait $! || true
  "$0" -m stageload.queue --near "$1" "$2"; rc=$?
  [ "$rc" -eq 0 ] && break
  if [ "$rc" -eq {NEAR_FAR} ]; then fails=0; else fails=$((fails + 1)); [ "$fails" -ge {NEAR_FAILS} ] && break; fi
done
trap - TERM INT HUP
STAGELOAD_TICKET_NS="$3" exec "$0" -m stageload.queue "$4"
"""


def near_head(lane_dir: Path, ticket: Path) -> bool:
    live = _live_tickets(lane_dir)
    if ticket not in dict(live):
        return True
    ahead, _ = lane_ahead(ticket, live, _last_starts(lane_dir), _active_hold(lane_dir),
                          lane_slots(lane_dir.name))
    return len(ahead) <= lane_slots(lane_dir.name) + NEAR_EXTRA


def _park(lane_dir: Path, ticket: Path, cmd: Sequence[str], config: GuardConfig, *,
          home: Path, lane: str, who: str, trace: Path | None, owner: str | None,
          deadline: str | None, gpu: bool, mem_peak: int | None, what: str, cwd: str | None,
          run_args: Sequence[str] | None) -> None:
    try:
        entry = json.loads(ticket.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        entry = {}
    deadline = entry.get("deadline") or deadline
    since = entry.get("since")
    spec = write_spec(cmd, config, home=home, lane=lane, who=who, trace=trace, owner=owner,
                      deadline=deadline, gpu=gpu, mem_peak=mem_peak, what=what, cwd=cwd,
                      run_args=run_args, since=since)
    if entry:
        entry["parked"] = _utc()
        try:
            _write_json(ticket, entry)
        except OSError:
            pass
    ns = ticket.name.split("-", 1)[0]
    py = sys.executable
    sys.stdout.flush()
    sys.stderr.flush()
    os.execv("/bin/sh", ["/bin/sh", "-c", _PARK_LOOP, py, str(lane_dir), str(ticket),
                         f"{int(ns)}:{who}", str(spec)])


def parked_ended(ticket: Path, spec: Path, code: int) -> None:
    """A parked loop got a signal: the history line the waiter would have written (state
    «waiting», exit 128 + the signal), and its spec and ticket are removed."""
    def read(path: Path) -> dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    kept, noted = read(ticket), read(spec)
    if kept or noted:
        who = str(kept.get("who") or noted.get("who") or "")
        # the ticket has every field of a history line; the spec stands in when it is gone
        entry = {"lane": ticket.parent.name, "who": who, "what": noted.get("what") or "",
                 "cmd": " ".join(noted.get("cmd") or [])[:300], "argv": noted.get("cmd"),
                 "cwd": noted.get("cwd"), "run_args": noted.get("run_args"),
                 "owner": owner_of(who, noted.get("owner")), "since": noted.get("since"), **kept}
        since = _timestamp(entry.get("since")) if entry.get("since") else 0.0
        waited = max(0.0, time.time() - since) if since else 0.0
        _write_history(ticket.parent.parent.parent, {
            **_record_of(entry), "exit": code, "state": "waiting", "waited_s": round(waited, 1),
            "run_s": 0.0, "peak_footprint": None, "paused_s": 0.0, "trace": None})
    spec.unlink(missing_ok=True)
    ticket.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--near":
        sys.exit(0 if near_head(Path(sys.argv[2]), Path(sys.argv[3])) else NEAR_FAR)
    if len(sys.argv) == 5 and sys.argv[1] == "--ended":
        parked_ended(Path(sys.argv[2]), Path(sys.argv[3]), int(sys.argv[4]))
        sys.exit(0)
    sys.exit(run_spec(Path(sys.argv[1])))
