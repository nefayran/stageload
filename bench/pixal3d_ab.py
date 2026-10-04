"""Eager against staged on one image: each run in its own guarded process, in turn."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

BUSY = (
    "ltx-2-mlx",
    "rife-ncnn-vulkan",
    "realcugan-ncnn-vulkan",
    "mflux",
    r"foley_mac\.py",
    r"generate_mps\.py",
    "MacOS/Blender -b",
    r"Godot\.app/Contents/MacOS/Godot",
    r"stageload\.pixal3d",
)


def run_one(python: str, image: Path, out: Path, mode: str, run: int, port: str | None) -> int:
    busy = [arg for pattern in BUSY for arg in ("--busy", pattern)]
    target = [
        python, "-m", "stageload.pixal3d", str(image), "-o", str(out / f"{mode}-{run}.glb"),
        "--load", mode, "--trace", str(out / f"{mode}-{run}.trace.jsonl"),
    ]
    if port:
        target += ["--port", port]
    cmd = [
        "nice", "-n", "15", python, "-m", "stageload", "guard", "--wait-free", "40",
        "--swap-budget", "8G", *busy, "--", "nice", "-n", "15", *target,
    ]
    with (out / f"{mode}-{run}.log").open("w") as log:
        return subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--port", default=None)
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in range(1, args.runs + 1):
        for mode in ("eager", "staged"):
            code = run_one(args.python, args.image, args.out, mode, run, args.port)
            runs.append({"mode": mode, "run": run, "exit": code})
            (args.out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if code != 0:
                print(f"{mode} run {run} exited with {code}; stopping. See {mode}-{run}.log.")
                return code
    print(f"done: {len(runs)} runs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
