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
    modes: dict[str, dict[str, list[Any]]] = {}
    for trace in sorted(results.glob("*.trace.jsonl")):
        s = summarize(read_trace(trace))
        mode = s["meta"].get("mode", trace.stem.split("-")[0])
        entry = modes.setdefault(
            mode,
            {"peak_footprint_gb": [], "swap_rise_gb": [], "seconds": [], "load_seconds": [],
             "stages": []},
        )
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
        "| mode | runs | peak footprint (GB) | swap rise (GB) | time (s) | of which loading (s) |",
        "|---|---|---|---|---|---|",
    ]
    for mode, entry in summary["modes"].items():

        def cell(key: str, entry: dict[str, list[Any]] = entry) -> str:
            return " / ".join(str(v) for v in entry[key])

        rows.append(
            f"| {mode} | {len(entry['seconds'])} | {cell('peak_footprint_gb')} | "
            f"{cell('swap_rise_gb')} | {cell('seconds')} | {cell('load_seconds')} |"
        )
    return "\n".join(rows) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    summary = collect(args.results)
    (args.results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.results / "table.md").write_text(table(summary))
    print(table(summary))


if __name__ == "__main__":
    main()
