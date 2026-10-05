"""Plot process footprint over time, one panel per trace, with the stages shaded."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "stageload"  # the same SVG ids on every run
import matplotlib.pyplot as plt  # noqa: E402

from stageload.trace import read_trace  # noqa: E402

LINE = {"eager": "#b4532a", "staged": "#2a6fb4"}
BAND = ("#f2f2f2", "#ffffff")


def _stage_windows(records: list[dict], end: float) -> list[tuple[str, float, float]]:
    starts = [(r["t"], r["stage"]) for r in records if r.get("kind") == "stage"]
    return [
        (stage, t0, starts[i + 1][0] if i + 1 < len(starts) else end)
        for i, (t0, stage) in enumerate(starts)
    ]


def _label(records: list[dict], path: Path) -> str:
    start = records[0] if records and records[0].get("type") == "start" else {}
    mode = start.get("mode", path.stem)
    if start.get("stop_at"):
        return f"{mode}, stopped where {start['stop_at']} begins"
    if not any(r.get("type") == "end" for r in records):
        return f"{mode}, stopped by the guard"
    return mode


def _ram_bytes() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


def plot(traces: list[Path], out: Path, ram_bytes: int | None = None) -> list[Path]:
    loaded = [read_trace(path) for path in traces]
    peak = max(r["footprint"] for records in loaded for r in records if r["type"] == "sample")
    ram_gb = (ram_bytes or _ram_bytes()) / 2**30
    span = max(r["t"] for records in loaded for r in records if r["type"] == "sample")
    fig, axes = plt.subplots(
        len(loaded), 1, figsize=(10, 2.8 * len(loaded)), sharex=True, sharey=True, squeeze=False
    )
    for ax, path, records in zip(axes[:, 0], traces, loaded, strict=True):
        mode = _label(records, path)
        samples = [r for r in records if r["type"] == "sample"]
        t = [s["t"] for s in samples]
        gb = [s["footprint"] / 2**30 for s in samples]
        windows = _stage_windows(records, t[-1])
        for i, (stage, t0, t1) in enumerate(windows):
            ax.axvspan(t0, t1, color=BAND[i % 2], zorder=0)
            # tiny stages stay unlabelled; the last one is where the run ended, so it always is
            if t1 - t0 >= span * 0.03 or i == len(windows) - 1:
                ax.text((t0 + t1) / 2, peak / 2**30 * 1.02, stage, ha="center", va="bottom",
                        fontsize=7, color="#555555")
        ax.axhline(ram_gb, color="#888888", linewidth=0.8, linestyle="--", zorder=1)
        ax.text(0, ram_gb, f" {ram_gb:.0f} GB of RAM", ha="left", va="bottom", fontsize=7,
                color="#666666")
        color = LINE.get(mode.split(",")[0], "#333333")
        ax.plot(t, gb, color=color, linewidth=1.6, zorder=2, label=mode)
        ax.set_ylabel("Footprint (GB)")
        ax.set_title(f"{mode}: peak {max(gb):.1f} GB", loc="left", fontsize=10)
        ax.grid(axis="y", color="#dddddd", linewidth=0.6)
        ax.set_ylim(0, peak / 2**30 * 1.25)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[-1, 0].set_xlabel("Time (s)")
    fig.tight_layout()
    written = []
    for suffix in (".svg", ".png"):
        target = out.with_suffix(suffix)
        # no date in the SVG: matplotlib writes the local time, and with it the time zone
        fig.savefig(target, dpi=160, metadata={"Date": None} if suffix == ".svg" else None)
        written.append(target)
    plt.close(fig)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("traces", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output path without extension")
    args = parser.parse_args()
    for path in plot(args.traces, args.out):
        print(path)


if __name__ == "__main__":
    main()
