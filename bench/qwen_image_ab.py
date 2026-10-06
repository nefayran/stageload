"""Qwen-Image-2.1 eager against staged: each run in its own guarded process, in turn.

Every run waits until no other heavy job of the machine is going (`--busy`, repeatable) and
the memory is free; runs alternate eager and staged so that a slower disk or a busier machine
does not favour one mode.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUSY = (r"qwen_image\.p[y] --mode", r"gen_qwen2[1]", r"generate_mps\.p[y]")
SCRIPT = Path(__file__).with_name("qwen_image.py")


def run_one(python: str, out: Path, mode: str, run: int, args: argparse.Namespace) -> int:
    busy = [arg for pattern in (*BUSY, *args.busy) for arg in ("--busy", pattern)]
    target = [
        python, str(SCRIPT), "--mode", mode, "--out", str(out / f"{mode}-{run}.png"),
        "--steps", str(args.steps), "--size", str(args.size), "--seed", str(args.seed),
    ]
    cmd = [
        "nice", "-n", "15", python, "-m", "stageload", "guard", "--wait-free", "40",
        "--swap-budget", "8G", "--quiet-for", args.quiet_for, "--start-timeout", args.start_timeout,
        *busy, "--", "nice", "-n", "15", *target,
    ]
    with (out / f"{mode}-{run}.log").open("w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--start-timeout", default="2h",
                        help="how long each run may wait for room and for other heavy jobs")
    parser.add_argument("--busy", action="append", default=[], metavar="REGEX",
                        help="also wait while a process matches this (repeatable)")
    parser.add_argument("--quiet-for", default="3m",
                        help="start a run only after no busy process was seen for this long")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in range(1, args.runs + 1):
        for mode in ("eager", "staged") if run % 2 else ("staged", "eager"):
            code = run_one(args.python, args.out, mode, run, args)
            runs.append({"mode": mode, "run": run, "exit": code})
            (args.out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if code != 0:
                print(f"{mode} run {run} exited with {code}; stopping. See {mode}-{run}.log.")
                return code
    print(f"done: {len(runs)} runs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
