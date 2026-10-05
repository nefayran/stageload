"""stageload-pixal3d: one image to a GLB with Pixal3D-mac, eager or staged, with a trace."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__
from .plan import STAGED_PIPELINES, STOP_STAGES
from .port import PortError, diagnostic_switches, find_port, load_port

# generate_mps.py flags that take main() down another branch or change the attention, by the
# name argparse gives them, so that an abbreviation such as --flash is caught too
REFUSED_FLAGS = {
    "flash_sdpa": "--flash-sdpa",
    "load_mesh": "--load-mesh",
    "load_fixture_07": "--load-fixture-07",
    "free_spent_models": "--free-spent-models",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stageload-pixal3d",
        description="Run Pixal3D-mac on one image with one stage's weights in memory at a time.",
        epilog="Any other flag is passed to generate_mps.py.",
        allow_abbrev=False,
    )
    parser.add_argument("image")
    parser.add_argument("-o", "--output", required=True, help="output .glb")
    parser.add_argument("--port", default=None, help="Pixal3D-mac checkout")
    parser.add_argument("--load", choices=("staged", "eager"), default="staged")
    parser.add_argument("--pipeline", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--trace", default=None, help="trace file (default: next to the output)")
    parser.add_argument("--stop-at", default=None, metavar="STAGE",
                        help="end the run where this stage would begin (no GLB): "
                        + ", ".join(STOP_STAGES))
    parser.add_argument("--fingerprint", default=None, metavar="DIR",
                        help="save a hash and a copy of every sampler's output here")
    return parser


def _fail(message: str) -> int:
    print(f"stageload-pixal3d: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    # parse_known_args, not a REMAINDER positional: a REMAINDER after `image` would swallow
    # every option written after the image, such as `photo.png -o out.glb`.
    a, extra = _parser().parse_known_args(argv)
    extra = [flag for flag in extra if flag != "--"]
    if a.stop_at is not None and a.stop_at not in STOP_STAGES:
        return _fail(f"--stop-at must be one of {', '.join(STOP_STAGES)}")
    if a.load == "staged":
        switches = diagnostic_switches()
        if switches:
            return _fail(f"{', '.join(switches)} only work with --load eager")
    try:
        port = load_port(find_port(a.port))
    except PortError as error:
        return _fail(str(error))
    args = port.gm.parse_args(
        [a.image, "--output", a.output, "--pipeline-type", a.pipeline, "--seed", str(a.seed),
         "--texture-size", str(a.texture_size), *extra]
    )
    refused = [flag for dest, flag in REFUSED_FLAGS.items() if getattr(args, dest, None)]
    if refused:
        return _fail(f"{', '.join(refused)} is not supported here; run generate_mps.py for it")
    if a.load == "staged" and args.pipeline_type not in STAGED_PIPELINES:
        return _fail(
            f"--load staged has a plan for {', '.join(STAGED_PIPELINES)} only; "
            f"use --load eager for {args.pipeline_type}"
        )
    port.gm._configure_fdg_environment(args)
    port.gm.load_runtime_deps()
    device = port.gm.resolve_device(args.device)

    import torch
    from PIL import Image

    from ..meter import MemoryMeter
    from .staged import generate

    output = Path(port.gm.output_glb_path(a.output))
    trace = Path(a.trace) if a.trace else output.with_suffix(".trace.jsonl")
    # from the port's arguments: a flag passed through to it overrides ours
    meta = {
        "mode": a.load,
        "pipeline": args.pipeline_type,
        "seed": args.seed,
        "texture_size": args.texture_size,
        "image": Path(args.image).name,
        "port_commit": port.commit,
        "port_dirty": port.dirty,
        "stageload": __version__,
        "torch": torch.__version__,
        "stop_at": a.stop_at,
    }
    image = Image.open(args.image)
    try:
        with MemoryMeter(trace, meta=meta) as meter:
            glb = generate(port, args, image, device, mode=a.load, sink=meter,
                           stop_at=a.stop_at, fingerprint_dir=a.fingerprint)
    except PortError as error:
        return _fail(str(error))
    summary = meter.summary()
    result = glb.name if glb is not None else f"stopped at {a.stop_at} as asked"
    print(
        f"stageload-pixal3d: {result} | {a.load} | peak footprint "
        f"{summary['peak_footprint'] / 2**30:.1f} GB | {summary['duration']:.0f} s | "
        f"trace {trace.name}"
    )
    return 0
