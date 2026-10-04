"""`stageload guard` and `stageload summary`."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="stageload")
    sub = parser.add_subparsers(dest="command", required=True)
    guard = sub.add_parser(
        "guard", help="run one command when there is room and stop it before swap runs away"
    )
    guard.add_argument(
        "--wait-free", type=int, default=40, metavar="PERCENT",
        help="memory available before starting (default 40)",
    )
    guard.add_argument(
        "--swap-budget", type=_arg(parse_size), default="8G",
        help="swap the command may add over what was used at its start (default 8G)",
    )
    guard.add_argument(
        "--swap-limit", type=_arg(parse_size), default=None,
        help="absolute swap ceiling, also checked before starting",
    )
    guard.add_argument(
        "--kill-free", type=int, default=10, metavar="PERCENT",
        help="stop the command after two samples under this (default 10)",
    )
    guard.add_argument(
        "--busy", action="append", default=[], metavar="REGEX",
        help="do not start while a process matches (repeatable)",
    )
    guard.add_argument("--start-timeout", type=_arg(parse_duration), default="30m")
    guard.add_argument("--poll", type=_arg(parse_duration), default="5s")
    guard.add_argument("cmd", nargs=argparse.REMAINDER, help="-- command ...")
    summary = sub.add_parser("summary", help="print the summary of a trace file as JSON")
    summary.add_argument("trace")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(json.dumps(summarize(read_trace(args.trace)), indent=2))
        return 0
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error("guard needs a command after --")
    config = GuardConfig(
        wait_free=args.wait_free,
        swap_budget=args.swap_budget,
        swap_limit=args.swap_limit,
        kill_free=args.kill_free,
        busy=tuple(args.busy),
        start_timeout=args.start_timeout,
        poll=args.poll,
    )
    return run_guarded(cmd, config)
