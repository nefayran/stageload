"""Bar chart of the peak footprint per stage, eager against staged, from a bench summary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "stageload"  # the same SVG ids on every run
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from stageload.trace import read_trace, summarize  # noqa: E402

COLORS = {"eager": "#b4532a", "staged": "#2a6fb4"}
INK, MUTED, SURFACE = "#222222", "#666666", "#fcfcfb"
GB = 2**30
SKIP = {"preprocess"}


def _mean_by_stage(runs: list[dict[str, float | None]]) -> dict[str, float]:
    stages: dict[str, list[float]] = {}
    for run in runs:
        for stage, value in run.items():
            if value is not None and stage not in SKIP:
                stages.setdefault(stage, []).append(value)
    return {stage: sum(values) / len(values) for stage, values in stages.items()}


def plot(summary_path: Path, out: Path, stopped_trace: Path | None = None) -> list[Path]:
    summary = json.loads(summary_path.read_text())
    eager = _mean_by_stage(summary["modes"]["eager"]["stages"])
    staged = _mean_by_stage(summary["modes"]["staged"]["stages"])
    stopped = None
    if stopped_trace is not None:
        stages = summarize(read_trace(stopped_trace))["stages"]
        stopped = next((s["peak_footprint"] / GB for s in stages if s["stage"] == "texture"), None)
    order = [stage for stage in staged if stage in eager or stage in staged]
    ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / GB

    fig, ax = plt.subplots(figsize=(10, 4.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.38
    for i, stage in enumerate(order):
        for offset, mode, values in ((-width / 2, "eager", eager), (width / 2, "staged", staged)):
            value = values.get(stage)
            hatched = False
            if value is None and mode == "eager" and stage == "texture" and stopped is not None:
                value, hatched = stopped, True
            if value is None:
                ax.text(i + offset, 1, "not\nreached", ha="center", va="bottom", fontsize=7,
                        color=MUTED)
                continue
            ax.bar(i + offset, value, width - 0.04, color=COLORS[mode] if not hatched else SURFACE,
                   edgecolor=COLORS[mode], hatch="////" if hatched else None, linewidth=1.2,
                   zorder=2)
            ax.text(i + offset, value + 0.8, f"{value:.1f}", ha="center", va="bottom",
                    fontsize=7.5, color=INK)
    ax.axhline(ram, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
    ax.text(-0.5, ram + 0.6, f"{ram:.0f} GB of RAM (the footprint also counts compressed memory)",
            fontsize=7.5, color=MUTED, va="bottom")
    ax.set_xticks(range(len(order)), order, color=INK)
    ax.set_ylabel("Peak footprint (GB)", color=INK)
    ax.set_title("Pixal3D 1024_cascade on a 48 GB Mac: peak footprint per stage", loc="left",
                 fontsize=11, color=INK, pad=22)
    ax.text(0, 1.02, "In the export no model weights are left; its peak comes from the port's "
            "mesh processing.", transform=ax.transAxes, fontsize=8, color=MUTED)
    ax.grid(axis="y", color="#e2e2e2", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=COLORS["eager"], label="eager"),
               Patch(facecolor=COLORS["staged"], label="staged")]
    if stopped is not None:
        handles.insert(1, Patch(facecolor=SURFACE, edgecolor=COLORS["eager"], hatch="////",
                                label="eager, stopped by the guard (2026-10-04 run)"))
    ax.legend(handles=handles, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.92),
              fontsize=9)
    ax.set_ylim(0, max([*eager.values(), *staged.values(), stopped or 0, ram]) * 1.18)
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, help="summary.json from summarize_runs.py")
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    parser.add_argument("--stopped-trace", type=Path, default=None,
                        help="trace of a full eager run the guard stopped in the texture stage")
    args = parser.parse_args()
    for path in plot(args.summary, args.out, args.stopped_trace):
        print(path)


if __name__ == "__main__":
    main()
