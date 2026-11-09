"""`stageload guard`, `run`, `queue`, `cancel`, `requeue`, `reload` and `summary`."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
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
        "--wait-free", type=_percent, default=35, metavar="PERCENT",
        help="memory available before starting (default 35, the start gate)",
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
        "--what", default=os.environ.get("STAGELOAD_WHAT", ""),
        help="what the run makes and what for, in plain English, shown by `stageload queue` and "
        "kept in history.jsonl (default $STAGELOAD_WHAT)",
    )
    run.add_argument(
        "--owner", default=os.environ.get("STAGELOAD_OWNER"),
        help="whose turn the run takes (default $STAGELOAD_OWNER, else --who up to its first dash)",
    )
    run.add_argument(
        "--deadline", default=os.environ.get("STAGELOAD_DEADLINE"),
        help="when what the run makes is due (ISO time, e.g. 2026-10-19T11:00Z; default $STAGELOAD_DEADLINE): "
        "an earlier deadline starts before a later one or none; turns stay fair among one deadline",
    )
    run.add_argument(
        "--gpu", action="store_true",
        default=os.environ.get("STAGELOAD_GPU", "").strip().lower() in ("1", "true", "yes"),
        help="the command computes on the GPU: it does not start while another --gpu command "
        "runs in any lane, and waits its turn for the GPU as within a lane "
        "(default $STAGELOAD_GPU=1)",
    )
    run.add_argument(
        "--lane", default="ram",
        help="commands of one lane run one at a time (ram: two at once when the budget fits, "
        "$STAGELOAD_RAM_SLOTS); lanes do not wait for each other. A comma list (win,light) "
        "waits in each and takes the first where the command can start; it gets the lane in "
        "$STAGELOAD_LANE (default ram)",
    )
    run.add_argument(
        "--mem-peak", type=_arg(parse_size), default=None, metavar="SIZE",
        help="the command's predicted peak memory, e.g. 25G. Without it: by the engine its "
        "command line names (MiniMax H3 video 25G, LTX 43G, Blender -b 4G, CUGAN/RIFE 3G, MMAudio/foley 8G, "
        "Whisper 3G) or 1.3 x the largest peak of the same --who (or of its numbered siblings) "
        "in history.jsonl, whichever "
        "is larger, else 16G in the ram lane and 4G in the others. Give it for a script that "
        "runs a heavy engine inside. A run starts only when the peaks of the running jobs of "
        "all lanes, its own and $STAGELOAD_RESERVE_GB (16) fit the machine's RAM",
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
    queue.add_argument("--shares", action="store_true", help="print each owner group's share and its use of the lane over 24 h")
    cancel = sub.add_parser(
        "cancel", help="end only the queued runs named by --who or --owner, waiting or running")
    named = cancel.add_mutually_exclusive_group(required=True)
    named.add_argument("--who", metavar="PATTERN", help="a glob on the run's --who (quote it)")
    named.add_argument("--owner", metavar="NAME", help="the run's owner, exactly")
    cancel.add_argument("--lane", help="only this lane")
    cancel.add_argument("--dry", action="store_true", help="only list what would be ended")
    requeue = sub.add_parser(
        "requeue", help="queue again the runs a signal ended (exit 143, 137, 130, -15) since a time")
    requeue.add_argument("--since", required=True, metavar="ISO",
                         help="runs that ended after this time, e.g. 2026-10-26T15:00Z")
    requeue.add_argument("--owner", metavar="NAME", help="only this owner's runs")
    requeue.add_argument("--dry", action="store_true", help="only print what would be started")
    reload = sub.add_parser(
        "reload", help="start the waiting runs again on the installed stageload, each in its place in the lane")
    reload.add_argument("--owner", metavar="NAME", help="only this owner's runs")
    reload.add_argument("--lane", help="only this lane")
    reload.add_argument("--dry", action="store_true", help="only list what would be started again")
    reload.add_argument("--end-for-retry", action="store_true",
                        help="also end the runs whose parent waits for them, keeping their place 10 min for the "
                        "script's own retry of the same --who (only for scripts that do retry)")
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


def _paused_for(row: dict[str, Any]) -> str:
    """How long a paused command has been paused, after its start time: «  paused 95 s»."""
    if row.get("state") != "paused":
        return ""
    try:
        since = datetime.fromisoformat(str(row["paused_since"]))
        return f"  paused {(datetime.now(timezone.utc) - since).total_seconds():.0f} s"
    except (KeyError, TypeError, ValueError):
        return "  paused"


def _what(row: dict[str, Any]) -> str:
    """The run's plain-English description. It comes after the fixed columns, so the readers
    of this line (the command center, the HQ office) still match it as before."""
    what = " ".join(str(row.get("what") or "").split())
    return f"  what: {what}" if what else ""


def _mem_wait(row: dict[str, Any]) -> str:
    """Why a run waits for memory, after its start time: «  memory: swap 8.3 GB > 6.0 GB»."""
    if row.get("state") != "mem-wait":
        return ""
    return f"  memory: {row.get('mem_wait') or 'checking'}"


def _show_queue(args: argparse.Namespace) -> int:
    from .queue import default_home, list_queue

    home = default_home()
    if args.shares:
        from .queue import share_report
        report = share_report(home, args.lane or "ram")
        if args.json:
            print(json.dumps(report, indent=2))
        elif not report:
            print("shares: none (no ~/.stageload/shares, no STAGELOAD_SHARES)")
        for r in report if not args.json else []:
            print(f"{r['group']:<10} share {r['share']:5.1f} %   used {r['used']:5.1f} %  ({r['used_h']:.2f} h of the last 24 h)")
        return 0
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
        who = (row['who'] or '-') + (" [gpu]" if row.get("gpu") else "")
        print(f"{row['lane']:6} {row['position']:>2} {row['state']:8} {who:20} "
              f"pid {row['pid']:<7} since {row['since']}{_paused_for(row)}{_mem_wait(row)}"
              f"{_what(row)}  {row['cmd'][:80]}")
    for record in history:
        print(f"done   {record['lane']:6} exit {record['exit']:<4} {record['who'] or '-':20} "
              f"waited {record['waited_s']:.0f} s, ran {record['run_s']:.0f} s, "
              f"peak {_gb(record['peak_footprint'])}  {record['cmd'][:60]}")
    return 0


def _cancel(args: argparse.Namespace) -> int:
    from .queue import cancel, default_home, matching_tickets

    home = default_home()
    rows = matching_tickets(home, who=args.who, owner=args.owner, lane=args.lane)
    named = f"--who {args.who!r}" if args.who is not None else f"--owner {args.owner!r}"
    if not rows:
        print(f"cancel: no queued run matches {named}"
              + (f" in lane {args.lane}" if args.lane else ""), file=sys.stderr)
        return 1
    if args.dry:
        for row in rows:
            print(f"would end {row['lane']:6} {row['state']:8} {row.get('who') or '-':20} "
                  f"pid {row['pid']:<7} {str(row.get('cmd') or '')[:60]}")
        return 0
    return 0 if cancel(home, rows) else 1


def _requeue(args: argparse.Namespace) -> int:
    from .queue import _since, default_home, requeue, requeue_plan

    try:
        since = _since(args.since)
    except ValueError:
        print(f"requeue: --since is not an ISO time: {args.since!r}", file=sys.stderr)
        return 2
    home = default_home()
    plan = requeue_plan(home, since, owner=args.owner)
    if not plan:
        print(f"requeue: no run ended by a signal since {args.since} is left to queue again")
        return 0
    requeue(home, plan, dry=args.dry)
    return 0


def _reload(args: argparse.Namespace) -> int:
    from .queue import default_home
    from .reload import reload, waiting_runs

    home = default_home()
    rows = waiting_runs(home, owner=args.owner, lane=args.lane)
    if not rows:
        print("reload: no waiting run in this Mac's lanes")
        return 0
    reload(home, rows, dry=args.dry, end_for_retry=args.end_for_retry)
    return 0


def _run_args(raw: Sequence[str], args: argparse.Namespace) -> list[str]:
    """The argv of `stageload run` itself, before `--`, with the values it took from the
    environment written out, so a requeue runs with the same ones."""
    tokens = list(raw[1:len(raw) - len(args.cmd)])
    if tokens[-1:] == ["--"]:  # an argparse that drops `--` from the remainder
        tokens.pop()

    def given(flag: str) -> bool:
        return any(t == flag or t.startswith(flag + "=") for t in tokens)

    for flag, value in (("--who", args.who), ("--what", args.what), ("--owner", args.owner),
                        ("--deadline", args.deadline)):
        if value and not given(flag):
            tokens += [flag, value]
    if args.gpu and not given("--gpu"):
        tokens.append("--gpu")
    return tokens


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "summary":
        print(json.dumps(summarize(read_trace(args.trace)), indent=2))
        return 0
    if args.command == "queue":
        return _show_queue(args)
    if args.command == "cancel":
        return _cancel(args)
    if args.command == "requeue":
        return _requeue(args)
    if args.command == "reload":
        return _reload(args)
    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        parser.error(f"{args.command} needs a command after --")
    if args.command == "guard":
        return run_guarded(cmd, _config(args))
    # imported here: a broken queue module must not take `stageload guard` down with it
    from .queue import default_home, default_trace, run_queued, write_spec

    home = default_home()
    trace = None if args.no_trace else (args.trace or default_trace(home, args.who))
    run_args = _run_args(list(sys.argv[1:] if argv is None else argv), args)
    try:
        cwd: str | None = os.getcwd()
    except OSError:
        cwd = None
    if argv is None:
        # started from a real command line: hand over to a process whose own command line does
        # not name the engine, or an older guard waiting for that engine takes the waiting run
        # for the engine itself, and the two wait for each other (2026-10-07). exec keeps the pid.
        spec = write_spec(cmd, _config(args), home=home, lane=args.lane, who=args.who, trace=trace,
                          owner=args.owner, deadline=args.deadline, gpu=args.gpu,
                          mem_peak=args.mem_peak, what=args.what, cwd=cwd, run_args=run_args)
        os.execv(sys.executable, [sys.executable, "-m", "stageload.queue", str(spec)])
    return run_queued(cmd, _config(args), home=home, lane=args.lane, who=args.who, trace=trace,
                      owner=args.owner, deadline=args.deadline, gpu=args.gpu,
                      mem_peak=args.mem_peak, what=args.what, cwd=cwd, run_args=run_args)
