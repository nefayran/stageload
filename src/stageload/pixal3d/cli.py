"""stageload-pixal3d: one image to a GLB with Pixal3D-mac, eager or staged, with a trace."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .. import __version__
from .port import PortError, diagnostic_switches, find_port, load_port

REFUSED_FLAGS = ("--flash-sdpa", "--load-mesh", "--load-fixture-07", "--free-spent-models")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stageload-pixal3d",
        description="Run Pixal3D-mac on one image with one stage's weights in memory at a time.",
        epilog="Any other flag is passed to generate_mps.py.",
    )
    parser.add_argument("image")
    parser.add_argument("-o", "--output", required=True, help="output .glb")
    parser.add_argument("--port", default=None, help="Pixal3D-mac checkout")
    parser.add_argument("--load", choices=("staged", "eager"), default="staged")
    parser.add_argument("--pipeline", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--trace", default=None, help="trace file (default: next to the output)")
    return parser


def _fail(message: str) -> int:
    print(f"stageload-pixal3d: {message}", file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    # parse_known_args, not a REMAINDER positional: a REMAINDER after `image` would swallow
    # every option written after the image, such as `photo.png -o out.glb`.
    a, extra = _parser().parse_known_args(argv)
    extra = [flag for flag in extra if flag != "--"]
    refused = [flag for flag in extra if flag.split("=", 1)[0] in REFUSED_FLAGS]
    if refused:
        return _fail(f"{', '.join(refused)} is not supported here; run generate_mps.py for it")
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
    port.gm._configure_fdg_environment(args)
    port.gm.load_runtime_deps()
    device = port.gm.resolve_device(args.device)

    import torch
    from PIL import Image

    from ..meter import MemoryMeter
    from .staged import generate

    output = Path(port.gm.output_glb_path(a.output))
    trace = Path(a.trace) if a.trace else output.with_suffix(".trace.jsonl")
    meta = {
        "mode": a.load,
        "pipeline": a.pipeline,
        "seed": a.seed,
        "texture_size": a.texture_size,
        "image": Path(a.image).name,
        "port_commit": port.commit,
        "port_dirty": port.dirty,
        "stageload": __version__,
        "torch": torch.__version__,
    }
    image = Image.open(a.image)
    with MemoryMeter(trace, meta=meta) as meter:
        glb = generate(port, args, image, device, mode=a.load, sink=meter)
    summary = meter.summary()
    print(
        f"stageload-pixal3d: {glb.name} | {a.load} | peak footprint "
        f"{summary['peak_footprint'] / 2**30:.1f} GB | {summary['duration']:.0f} s | "
        f"trace {trace.name}"
    )
    return 0
