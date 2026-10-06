"""Bar chart of the peak footprint per stage of the Qwen-Image bench, from its summary.

The eager run the guard stopped is drawn hatched in the stage it was stopped in: its bar is the
footprint when the guard stopped it, not the peak the stage would have reached.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from matplotlib.patches import Patch
from plot_stages import COLORS, INK, MUTED, SURFACE, plt  # also sets the backend and SVG ids

LABELS = {"setup": "load", "encode": "encode", "denoise": "denoise", "decode": "decode"}


def _mean(runs: list[dict[str, float | None]], stage: str) -> float | None:
    values = [run[stage] for run in runs if run.get(stage) is not None]
    return sum(values) / len(values) if values else None


def plot(summary_path: Path, out: Path) -> list[Path]:
    summary = json.loads(summary_path.read_text())
    eager, staged = summary["modes"]["eager"], summary["modes"]["staged"]
    stopped = {stop for stop in eager["stopped_at"] if stop}
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.38
    order = list(LABELS)
    for i, stage in enumerate(order):
        for offset, mode, runs in ((-width / 2, "eager", eager), (width / 2, "staged", staged)):
            value = _mean(runs["stages"], stage)
            if value is None:
                continue
            hatched = mode == "eager" and stage in stopped
            ax.bar(i + offset, value, width - 0.04, color=SURFACE if hatched else COLORS[mode],
                   edgecolor=COLORS[mode], hatch="////" if hatched else None, linewidth=1.2,
                   zorder=2)
            ax.text(i + offset, value + 0.8, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.axhline(ram, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(len(order) - 0.45, ram + 0.6,
            f"{ram:.0f} GB of RAM (the footprint also counts compressed memory)",
            fontsize=7.5, color=MUTED, va="bottom", ha="right")
    ax.set_xticks(range(len(order)), [LABELS[s] for s in order], color=INK)
    ax.set_ylabel("Peak footprint (GB)", color=INK)
    ax.set_title("Qwen-Image-2.1 (diffusers) on a 48 GB Mac, 1024 px, 20 steps: "
                 "peak footprint per stage", loc="left", fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "Staged keeps the VAE and loads the text encoder (16.3 GB) and the "
            "transformer (13.3 GB) one at a time.", transform=ax.transAxes, fontsize=8,
            color=MUTED)
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=COLORS["eager"], label="eager"),
               Patch(facecolor=SURFACE, edgecolor=COLORS["eager"], hatch="////",
                     label="eager, stopped by the guard (swap +8 GB)"),
               Patch(facecolor=COLORS["staged"], label=f"staged (mean of {len(staged['seconds'])} "
                     "runs)")]
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.80),
              fontsize=9)
    ax.set_ylim(0, ram * 1.2)
    fig.tight_layout()
    written = []
    for suffix in (".svg", ".png"):
        target = out.with_suffix(suffix)
        # no date in the SVG: matplotlib writes the local time, and with it the time zone
        fig.savefig(target, dpi=170, facecolor=SURFACE,
                    metadata={"Date": None} if suffix == ".svg" else None)
        written.append(target)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("summary", type=Path, help="summary.json from summarize_runs.py")
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = parser.parse_args()
    for path in plot(args.summary, args.out):
        print(path)


if __name__ == "__main__":
    main()
