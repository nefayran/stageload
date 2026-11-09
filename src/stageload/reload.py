"""`stageload reload`: waiting runs started again on the stageload installed now, each in its place in its lane.

A process keeps the code it started with, so after an upgrade the runs already waiting follow the old rules. For each
waiting run on this machine's lanes, oldest first:

- a run whose parent is launchd (started detached) is started again from its ticket (run_args and argv);
- a run whose parent is a wrapper named in $STAGELOAD_RELOAD_WRAPPERS, itself started detached, has the wrapper started
  again with its own argv, since the wrapper waits for the run's exit code and then handles its output;
- any other run is left alone and listed: its parent waits for its exit code, and a SIGTERM would look like a failure.
  With --end-for-retry, for scripts that queue the same --who again when a run ends before it starts, such a run gets
  SIGTERM and its ticket time is kept in places/<who>.json for 10 min; the next `stageload run` of that who takes it.

The restarted process gets the old one's environment as the kernel keeps it (KERN_PROCARGS2), its cwd and its stdout
file, plus STAGELOAD_TICKET_NS=«ns:who», so the new ticket of that who takes the old ticket's time and sorts where it
stood. The old processes get SIGTERM first, the wrapper before its run, so the wrapper never reports the run's exit and
no two processes write one log; the few seconds between are the only time the ticket is off the lane. A «reload» line
goes to cancelled.jsonl, so `stageload requeue` never sends the old run again. Running and paused runs are never touched.
"""
from __future__ import annotations

import ctypes
import json
import os
import signal
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .queue import _all_tickets, _ended, _owner, _utc, asker, keep_place, remote_lanes

WAITING = ("waiting", "mem-wait", "gpu-wait")


def wrappers() -> dict[str, tuple[str, ...]]:
    """$STAGELOAD_RELOAD_WRAPPERS: scripts that only prepare a run, wait for it and handle its output, so they can be
    started again whole. Comma-separated `name.py` or `name.py:--flag:--other`; a call passing one of the flags is
    skipped, since it queues another run first and that run would be made again."""
    out = {}
    for item in os.environ.get("STAGELOAD_RELOAD_WRAPPERS", "").split(","):
        name, *flags = [p.strip() for p in item.split(":")]
        if name:
            out[name] = tuple(f for f in flags if f)
    return out


def procargs(pid: int) -> tuple[list[str], dict[str, str]] | None:
    """argv and environment of ``pid`` as the kernel keeps them (macOS sysctl KERN_PROCARGS2), or None."""
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL(None, use_errno=True)
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2
    size = ctypes.c_size_t(0)
    if libc.sysctl(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        return None
    buf = ctypes.create_string_buffer(size.value)
    if libc.sysctl(mib, 3, buf, ctypes.byref(size), None, 0) != 0:
        return None
    raw = buf.raw[:size.value]
    argc = struct.unpack_from("i", raw)[0]
    rest = raw[4:]
    pos = rest.find(b"\0")          # the executable path, then NUL padding
    while 0 <= pos < len(rest) and rest[pos] == 0:
        pos += 1
    parts = rest[pos:].split(b"\0")
    argv = [p.decode("utf-8", "replace") for p in parts[:argc]]
    env = {}
    for p in parts[argc:]:
        if not p:
            break
        key, sep, value = p.decode("utf-8", "replace").partition("=")
        if sep:
            env[key] = value
    return argv, env


def _ps(pid: int, field: str) -> str:
    return subprocess.run(["ps", "-o", f"{field}=", "-p", str(pid)], capture_output=True, text=True,
                          check=False).stdout.strip()


def _fd_file(pid: int, fd: str) -> str | None:
    """The path open as ``fd`` (1 for stdout, cwd) in ``pid``, or None."""
    out = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", fd, "-Fn"], capture_output=True, text=True,
                         check=False).stdout
    names = [line[1:] for line in out.splitlines() if line.startswith("n")]
    return names[0] if names else None


def waiting_runs(home: Path, owner: str | None = None, lane: str | None = None) -> list[dict[str, Any]]:
    """One row per waiting run with a ticket in a lane of this Mac: pid, its ticket time, lanes, who, owner."""
    remote, rows = remote_lanes(), {}
    for path, data in _all_tickets(home):
        if path.parent.name in remote or (lane is not None and path.parent.name != lane):
            continue
        if owner is not None and _owner(data) != owner:
            continue
        ns, pid = path.stem.split("-")[0], int(data.get("pid") or path.stem.split("-")[1])
        row = rows.setdefault(pid, {"pid": pid, "ns": ns, "lanes": [], "who": data.get("who") or "",
                                    "owner": _owner(data), "states": set(), "cwd": data.get("cwd"),
                                    "argv": data.get("argv"), "run_args": data.get("run_args")})
        row["lanes"].append(path.parent.name)
        row["states"].add(data.get("state", "waiting"))
    return [r for r in rows.values() if r["states"] <= set(WAITING)]


def plan(row: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """How to start ``row`` again: {kill, argv, env, cwd, out}, or None and why not."""
    pid = row["pid"]
    ppid = _ps(pid, "ppid")
    if not ppid.isdigit():
        return None, "it is gone"
    if ppid == "1":
        if row.get("run_args") is None or not row.get("argv"):
            return None, "its ticket has no run_args (queued before 0542618)"
        got = procargs(pid)
        if got is None or not got[0] or not row.get("cwd") or not Path(row["cwd"]).is_dir():
            return None, "its environment or cwd cannot be read"
        out = _fd_file(pid, "1")
        return {"kill": [pid], "argv": [got[0][0], "-m", "stageload", "run", *row["run_args"], "--", *row["argv"]],
                "env": got[1], "cwd": row["cwd"], "out": out if out and Path(out).is_file() else None}, ""
    parent = int(ppid)
    got = procargs(parent)
    known = wrappers()
    wrapper = next((w for w in known if got and any(Path(a).name == w for a in got[0])), None)
    if got is None or wrapper is None:
        return None, f"its parent {parent} ({_ps(parent, 'command')[:50]}) waits for its exit"
    if any(flag in got[0] for flag in known[wrapper]):
        return None, f"its wrapper {wrapper} {' '.join(known[wrapper])} would queue another run first"
    if _ps(parent, "ppid") != "1":
        return None, f"its wrapper {wrapper} {parent} has a parent that waits for it"
    cwd, out = _fd_file(parent, "cwd"), _fd_file(parent, "1")
    if not cwd or not Path(cwd).is_dir():
        return None, f"the cwd of its wrapper {parent} cannot be read"
    return {"kill": [parent, pid], "argv": got[0], "env": got[1], "cwd": cwd,
            "out": out if out and Path(out).is_file() else None}, ""


def _ticket_of(home: Path, ns: str, old_pid: int) -> int | None:
    for p in (home / "queue").glob(f"*/{ns}-*.json"):
        pid = int(p.stem.split("-")[1])
        if pid != old_pid:
            return pid
    return None


def reload(home: Path, rows: list[dict[str, Any]], dry: bool = False, wait: float = 60.0,
           log: Callable[[str], None] = print, end_for_retry: bool = False) -> list[dict[str, Any]]:
    """Start each run of ``rows`` again on the installed code, in its place (see the module doc)."""
    done, who_asked = [], asker()
    for row in sorted(rows, key=lambda r: r["ns"]):
        head = f"{','.join(row['lanes']):10} {row['who'][:28]:28} pid {row['pid']:<6}"
        how, why = plan(row)
        if how is None and end_for_retry and "waits for" in why and row["who"]:
            if dry:
                log(f"would end {head} for its script to queue again, its place kept ({why})")
                continue
            keep_place(home, row["who"], row["ns"])
            try:
                os.kill(row["pid"], signal.SIGTERM)
            except ProcessLookupError:
                continue
            line = {"utc": _utc(), "who": row["who"], "owner": row["owner"], "lane": ",".join(row["lanes"]),
                    "state": "/".join(sorted(row["states"])), "pid": row["pid"], "reason": "reload-for-retry",
                    **who_asked}
            with (home / "cancelled.jsonl").open("a", encoding="utf-8") as file:
                file.write(json.dumps(line) + "\n")
            log(f"ended    {head} for its script to queue again; its place is kept 10 min")
            done.append(line)
            continue
        if how is None:
            log(f"left     {head} {why}")
            continue
        if dry:
            log(f"would    {head} {' '.join(how['argv'])[:90]}")
            continue
        for pid in how["kill"]:                       # the wrapper first: it must not see the run end
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        until = time.monotonic() + 15
        while time.monotonic() < until and not all(_ended(p) for p in how["kill"]):
            time.sleep(0.05)
        out_path = how["out"]
        if out_path is None:
            (home / "reloaded").mkdir(parents=True, exist_ok=True)
            out_path = str(home / "reloaded" / f"{row['who'].replace('/', '_') or 'job'}-{row['ns']}.log")
        env = {**how["env"], "STAGELOAD_TICKET_NS": f"{row['ns']}:{row['who']}", "STAGELOAD_HOME": str(home)}
        with open(out_path, "ab") as out:
            out.write(f"\nstageload reload {_utc()}: started again on the installed stageload, old pid "
                      f"{how['kill'][0]}\n".encode())
            out.flush()
            new = subprocess.Popen(how["argv"], cwd=how["cwd"], env=env, start_new_session=True,
                                   stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT)
        until, run_pid = time.monotonic() + wait, None
        while time.monotonic() < until and new.poll() is None and run_pid is None:
            run_pid = _ticket_of(home, row["ns"], row["pid"])
            time.sleep(0.1)
        run_pid = run_pid or _ticket_of(home, row["ns"], row["pid"])
        line = {"utc": _utc(), "who": row["who"], "owner": row["owner"], "lane": ",".join(row["lanes"]),
                "state": "/".join(sorted(row["states"])), "pid": row["pid"], "reason": "reload",
                "new_pid": new.pid, **who_asked}
        with (home / "cancelled.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(line) + "\n")
        if run_pid is None:
            log(f"FAILED   {head} the new process {new.pid} put no ticket in its place "
                f"(exit {new.poll()}); see {out_path}")
            continue
        log(f"reloaded {head} as pid {new.pid}" + (f", its run {run_pid}" if run_pid != new.pid else ""))
        done.append(line)
    return done
