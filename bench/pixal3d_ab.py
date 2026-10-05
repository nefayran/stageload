"""Eager against staged on one image: each run in its own guarded process, in turn.

`--eager-stop-at texture` ends eager runs where the texture stage would begin, for a machine
where eager loading does not fit; `--fingerprints` saves every sampler's output for
compare_fingerprints.py.
"""

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


def run_one(python: str, image: Path, out: Path, mode: str, run: int, port: str | None,
            stop_at: str | None = None, fingerprints: bool = False,
            start_timeout: str = "30m") -> int:
    busy = [arg for pattern in BUSY for arg in ("--busy", pattern)]
    target = [
        python, "-m", "stageload.pixal3d", str(image), "-o", str(out / f"{mode}-{run}.glb"),
        "--load", mode, "--trace", str(out / f"{mode}-{run}.trace.jsonl"),
    ]
    if port:
        target += ["--port", port]
    if stop_at:
        target += ["--stop-at", stop_at]
    if fingerprints:
        target += ["--fingerprint", str(out / f"fp-{mode}-{run}")]
    cmd = [
        "nice", "-n", "15", python, "-m", "stageload", "guard", "--wait-free", "40",
        "--swap-budget", "8G", "--start-timeout", start_timeout, *busy,
        "--", "nice", "-n", "15", *target,
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
    parser.add_argument("--eager-stop-at", default=None, metavar="STAGE")
    parser.add_argument("--fingerprints", action="store_true")
    parser.add_argument("--start-timeout", default="30m",
                        help="how long each run may wait for room and for other heavy jobs")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    runs = []
    for run in range(1, args.runs + 1):
        for mode in ("eager", "staged"):
            stop_at = args.eager_stop_at if mode == "eager" else None
            code = run_one(args.python, args.image, args.out, mode, run, args.port, stop_at,
                           args.fingerprints, args.start_timeout)
            runs.append({"mode": mode, "run": run, "exit": code})
            (args.out / "runs.json").write_text(json.dumps(runs, indent=2) + "\n")
            if code != 0:
                print(f"{mode} run {run} exited with {code}; stopping. See {mode}-{run}.log.")
                return code
    print(f"done: {len(runs)} runs in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
