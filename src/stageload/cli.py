"""`stageload guard`, `stageload run`, `stageload queue` and `stageload summary`."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .guard import GuardConfig, run_guarded
from .trace import read_trace, summarize
from .units import parse_duration, parse_size


def _arg(parse: Callable[[str], Any]) -> Callable[[str], Any]:
    def convert(text: str) -> Any:
        try:
            return parse(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(str(error)) from error

    return convert


def _percent(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a whole percent: {text!r}") from error
    if not 0 <= value <= 100:
        raise argparse.ArgumentTypeError(f"percent out of range 0-100: {value}")
    return value


def _positive_duration(text: str) -> float:
    value = _arg(parse_duration)(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be more than zero: {text!r}")
    return value


def _regex(text: str) -> str:
    try:
        re.compile(text)
    except re.error as error:
        raise argparse.ArgumentTypeError(f"bad regex {text!r}: {error}") from error
    return text


def _memory_flags(parser: argparse.ArgumentParser, *, quiet_for: str, start_timeout: str) -> None:
    """The flags `guard` and `run` share, so a script moves from one to the other by one word."""
    parser.add_argument(
        "--wait-free", type=_percent, default=40, metavar="PERCENT",
        help="memory available before starting (default 40)",
    )
    parser.add_argument(
        "--swap-budget", type=_arg(parse_size), default="8G",
        help="swap the command may add over what was used at its start (default 8G)",
    )
    parser.add_argument(
        "--swap-limit", type=_arg(parse_size), default=None,
        help="absolute swap ceiling, also checked before starting",
    )
    parser.add_argument(
        "--kill-free", type=_percent, default=10, metavar="PERCENT",
        help="stop the command after two samples under this (default 10)",
    )
    parser.add_argument(
        "--busy", action="append", type=_regex, default=[], metavar="REGEX",
        help="do not start while a process matches (repeatable)",
    )
    parser.add_argument(
        "--quiet-for", type=_arg(parse_duration), default=quiet_for,
        help=f"with --busy: start only after no busy process was seen for this long "
        f"(default {quiet_for})",
    )
    parser.add_argument("--start-timeout", type=_arg(parse_duration), default=start_timeout)
    parser.add_argument("--poll", type=_positive_duration, default="5s")
    parser.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command ...")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stageload")
    sub = parser.add_subparsers(dest="command", required=True)
    guard = sub.add_parser(
        "guard", help="run one command when there is room and stop it before swap runs away"
    )
    _memory_flags(guard, quiet_for="1m", start_timeout="30m")
    run = sub.add_parser(
        "run",
        help="queue a heavy command behind the others of its lane, then run it like guard, "
        "metered",
    )
    _memory_flags(run, quiet_for="20s", start_timeout="24h")
    run.add_argument(
        "--who", default=os.environ.get("STAGELOAD_WHO", ""),
        help="who queued it, shown by `stageload queue` (default $STAGELOAD_WHO)",
    )
    run.add_argument(
        "--owner", default=os.environ.get("STAGELOAD_OWNER"),
        help="whose turn the run takes (default $STAGELOAD_OWNER, else --who up to its first dash)",
    )
    run.add_argument(
        "--lane", default="ram",
        help="commands of one lane run one at a time; lanes do not wait for each other "
        "(default ram)",
    )
    traced = run.add_mutually_exclusive_group()
    traced.add_argument("--trace", type=Path, help="where to write the memory trace")
    traced.add_argument("--no-trace", action="store_true", help="do not meter the command")
    queue = sub.add_parser("queue", help="show the running and waiting commands")
    queue.add_argument("--lane", help="only this lane")
    queue.add_argument("--json", action="store_true", help="print JSON")
    queue.add_argument(
        "--history", type=int, default=0, metavar="N", help="also show the last N finished runs"
    )
    summary = sub.add_parser("summary", help="print the summary of a trace file as JSON")
    summary.add_argument("trace")
    return parser


def _config(args: argparse.Namespace) -> GuardConfig:
    return GuardConfig(
        wait_free=args.wait_free,
        swap_budget=args.swap_budget,
        swap_limit=args.swap_limit,
        kill_free=args.kill_free,
        busy=tuple(args.busy),
        quiet_for=args.quiet_for,
        start_timeout=args.start_timeout,
        poll=args.poll,
    )


def _gb(nbytes: int | None) -> str:
    return "-" if not nbytes else f"{nbytes / 2**30:.1f} GB"


def _show_queue(args: argparse.Namespace) -> int:
    from .queue import default_home, list_queue

    home = default_home()
    rows = list_queue(home, args.lane)
    history = []
    if args.history > 0 and (home / "history.jsonl").exists():
        lines = (home / "history.jsonl").read_text(encoding="utf-8").splitlines()
        history = [json.loads(line) for line in lines[-args.history :] if line.strip()]
    if args.json:
        print(json.dumps({"queue": rows, "history": history} if args.history else rows, indent=2))
        return 0
    if not rows:
        print("queue: empty")
    for row in rows:
        print(f"{row['lane']:6} {row['position']:>2} {row['state']:8} {row['who'] or '-':20} "
              f"pid {row['pid']:<7} since {row['since']}  {row['cmd'][:80]}")
    for record in history:
        print(f"done   {record['lane']:6} exit {record['exit']:<4} {record['who'] or '-':20} "
              f"waited {record['waited_s']:.0f} s, ran {record['run_s']:.0f} s, "
              f"peak {_gb(record['peak_footprint'])}  {record['cmd'][:60]}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(json.dumps(summarize(read_trace(args.trace)), indent=2))
        return 0
    if args.command == "queue":
        return _show_queue(args)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error(f"{args.command} needs a command after --")
    if args.command == "guard":
        return run_guarded(cmd, _config(args))
    # imported here: a broken queue module must not take `stageload guard` down with it
    from .queue import default_home, default_trace, run_queued, write_spec

    home = default_home()
    trace = None if args.no_trace else (args.trace or default_trace(home, args.who))
    if argv is None:
        # started from a real command line: hand over to a process whose own command line does
        # not name the engine, or an older guard waiting for that engine takes the waiting run
        # for the engine itself, and the two wait for each other (2026-10-07). exec keeps the pid.
        spec = write_spec(cmd, _config(args), home=home, lane=args.lane, who=args.who, trace=trace,
                          owner=args.owner)
        os.execv(sys.executable, [sys.executable, "-m", "stageload.queue", str(spec)])
    return run_queued(cmd, _config(args), home=home, lane=args.lane, who=args.who, trace=trace,
                      owner=args.owner)
