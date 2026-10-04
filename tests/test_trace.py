from stageload.trace import summarize


def sample(t, footprint, swap=0, level=80):
    return {"type": "sample", "t": t, "footprint": footprint, "swap_used": swap, "level": level}


def event(t, kind, **fields):
    return {"type": "event", "t": t, "kind": kind, **fields}


RECORDS = [
    {"type": "start", "utc": "2026-10-04T10:00:00+00:00", "interval": 0.5, "mode": "staged"},
    sample(0.0, 100, swap=10),
    event(0.1, "stage", stage="structure"),
    event(0.2, "load", name="ss_flow", nbytes=50, seconds=1.0, stage="structure"),
    sample(1.0, 300, swap=20, level=60),
    event(2.0, "stage", stage="texture"),
    event(2.1, "release", name="ss_flow", nbytes=50, stage="texture"),
    event(2.2, "load", name="tex", nbytes=70, seconds=2.0, stage="texture", note="outside_stage"),
    sample(3.0, 250, swap=15, level=70),
    sample(4.0, 120, swap=12),
    {"type": "end"},
]


def test_summary_reports_peaks_swap_and_loads():
    s = summarize(RECORDS)
    assert s["meta"] == {"utc": "2026-10-04T10:00:00+00:00", "interval": 0.5, "mode": "staged"}
    assert s["duration"] == 4.0
    assert s["peak_footprint"] == 300
    assert s["swap"] == {"start": 10, "peak": 20, "end": 12}
    assert s["min_level"] == 60
    assert s["loads"] == {"count": 2, "bytes": 120, "seconds": 3.0}
    assert s["releases"] == {"count": 1, "bytes": 50}
    assert s["outside_stage"] == ["tex"]


def test_stages_cover_the_run_in_order_with_their_own_peaks():
    stages = summarize(RECORDS)["stages"]
    assert stages == [
        {"stage": "setup", "seconds": 0.1, "peak_footprint": 100},
        {"stage": "structure", "seconds": 1.9, "peak_footprint": 300},
        {"stage": "texture", "seconds": 2.0, "peak_footprint": 250},
    ]


def test_an_empty_trace_summarizes_to_zeros():
    s = summarize([])
    assert s["peak_footprint"] == 0
    assert s["stages"] == []
