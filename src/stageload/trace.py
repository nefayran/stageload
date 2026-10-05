"""Read a stageload trace and summarize it."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_trace(path: str | Path) -> list[dict[str, Any]]:
    """Records of a trace file. A half-written last line, left by a killed run, is skipped."""
    lines = [line for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise
    return records


def _windows(stage_events: list[dict[str, Any]], end: float) -> list[tuple[str, float, float]]:
    """Consecutive (stage, start, end) windows; the time before the first stage is ``setup``.

    Entering the stage the run is already in does not start a new window.
    """
    starts = [(0.0, "setup")] + [(e["t"], e["stage"]) for e in stage_events]
    if len(starts) > 1 and starts[1][0] <= 0.0:
        starts = [(0.0, starts[1][1])] + starts[2:]
    starts = [s for i, s in enumerate(starts) if i == 0 or s[1] != starts[i - 1][1]]
    windows = []
    for i, (t0, stage) in enumerate(starts):
        t1 = starts[i + 1][0] if i + 1 < len(starts) else end
        windows.append((stage, t0, t1))
    return windows


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    start = next((r for r in records if r["type"] == "start"), {})
    samples = [r for r in records if r["type"] == "sample"]
    events = [r for r in records if r["type"] == "event"]
    loads = [e for e in events if e["kind"] in ("load", "reload")]
    releases = [e for e in events if e["kind"] == "release"]
    end = samples[-1]["t"] if samples else 0.0
    stages = []
    if samples:
        for stage, t0, t1 in _windows([e for e in events if e["kind"] == "stage"], end):
            inside = [s["footprint"] for s in samples if t0 <= s["t"] < t1 or s["t"] == t1 == end]
            stages.append(
                {
                    "stage": stage,
                    "seconds": round(t1 - t0, 3),
                    "peak_footprint": max(inside) if inside else None,
                }
            )
    swap = (
        {
            "start": samples[0]["swap_used"],
            "peak": max(s["swap_used"] for s in samples),
            "end": samples[-1]["swap_used"],
        }
        if samples
        else {}
    )
    return {
        "meta": {key: value for key, value in start.items() if key != "type"},
        "duration": round(end - samples[0]["t"], 3) if samples else 0.0,
        "peak_footprint": max((s["footprint"] for s in samples), default=0),
        "swap": swap,
        "min_level": min((s["level"] for s in samples), default=None),
        "stages": stages,
        "loads": {
            "count": len(loads),
            "bytes": sum(e.get("nbytes", 0) for e in loads),
            "seconds": round(sum(e.get("seconds", 0.0) for e in loads), 3),
        },
        "releases": {"count": len(releases), "bytes": sum(e.get("nbytes", 0) for e in releases)},
        "outside_stage": sorted({e["name"] for e in loads if e.get("note") == "outside_stage"}),
    }
