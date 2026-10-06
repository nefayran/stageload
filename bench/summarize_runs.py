"""Collect the traces of a bench run into summary.json and a Markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from stageload.trace import read_trace, summarize

GB = 2**30


def collect(results: Path) -> dict[str, Any]:
    runs = json.loads((results / "runs.json").read_text())
    exits = {f"{r['mode']}-{r['run']}": r["exit"] for r in runs}
    modes: dict[str, dict[str, list[Any]]] = {}
    for trace in sorted(results.glob("*.trace.jsonl")):
        s = summarize(read_trace(trace))
        mode = s["meta"].get("mode", trace.stem.split("-")[0])
        stopped = s["meta"].get("stop_at")
        if stopped is None and exits.get(trace.name.removesuffix(".trace.jsonl"), 0) != 0:
            # stopped from outside, by the guard: in the stage its trace ends in
            stopped = s["stages"][-1]["stage"] if s["stages"] else "start"
        entry = modes.setdefault(
            mode,
            {"peak_footprint_gb": [], "swap_rise_gb": [], "seconds": [], "load_seconds": [],
             "stopped_at": [], "stages": []},
        )
        entry["stopped_at"].append(stopped)
        entry["peak_footprint_gb"].append(round(s["peak_footprint"] / GB, 2))
        entry["swap_rise_gb"].append(round((s["swap"]["peak"] - s["swap"]["start"]) / GB, 2))
        entry["seconds"].append(round(s["duration"], 1))
        entry["load_seconds"].append(round(s["loads"]["seconds"], 1))
        entry["stages"].append(
            {
                st["stage"]: None if st["peak_footprint"] is None
                else round(st["peak_footprint"] / GB, 2)
                for st in s["stages"]
            }
        )
    comparisons = {}
    for path in sorted(results.glob("compare-*.json")):
        comparisons[path.stem.removeprefix("compare-")] = json.loads(path.read_text())
    return {"runs": runs, "modes": modes, "comparisons": comparisons}


def table(summary: dict[str, Any]) -> str:
    rows = [
        "| mode | runs | ran to | peak footprint (GB) | swap rise (GB) | time (s) "
        "| of which loading (s) |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode, entry in summary["modes"].items():

        def cell(key: str, entry: dict[str, list[Any]] = entry) -> str:
            return " / ".join(str(v) for v in entry[key])

        ran_to = " / ".join(f"{stop} (stopped)" if stop else "end" for stop in entry["stopped_at"])
        rows.append(
            f"| {mode} | {len(entry['seconds'])} | {ran_to} | {cell('peak_footprint_gb')} | "
            f"{cell('swap_rise_gb')} | {cell('seconds')} | {cell('load_seconds')} |"
        )
    return "\n".join(rows) + "\n"


def stage_table(summary: dict[str, Any]) -> str:
    """Peak footprint per stage, one column per mode, the runs of a mode separated by '/'."""
    modes = list(summary["modes"])
    order: list[str] = []
    for mode in modes:
        for stages in summary["modes"][mode]["stages"]:
            order += [stage for stage in stages if stage not in order]
    rows = ["| stage | " + " | ".join(f"{m} (GB)" for m in modes) + " |",
            "|---|" + "---|" * len(modes)]
    for stage in order:
        cells = []
        for mode in modes:
            values = [runs.get(stage) for runs in summary["modes"][mode]["stages"]]
            cells.append(" / ".join("–" if v is None else str(v) for v in values))
        rows.append(f"| {stage} | " + " | ".join(cells) + " |")
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary = collect(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    text = table(summary) + "\n" + stage_table(summary)
    (args.results / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
