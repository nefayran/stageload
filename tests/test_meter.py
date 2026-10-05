import json
import sys
import time

import pytest

from stageload import Event
from stageload.meter import MemoryMeter, Readers

macos = pytest.mark.skipif(sys.platform != "darwin", reason="macOS memory counters")


def fake_readers(values):
    it = iter(values)
    last = [values[0]]

    def footprint():
        try:
            last[0] = next(it)
        except StopIteration:
            pass
        return last[0]

    return Readers(footprint=footprint, swap_used=lambda: 5, level=lambda: 77)


def test_the_meter_writes_samples_and_events_as_json_lines(tmp_path):
    path = tmp_path / "trace.jsonl"
    readers = fake_readers([1, 9, 3])
    with MemoryMeter(path, interval=0.01, meta={"mode": "test"}, readers=readers) as meter:
        meter.emit(Event("stage", stage="one"))
        time.sleep(0.05)
        meter.mark("halfway")
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["type"] == "start" and lines[0]["mode"] == "test"
    assert lines[-1] == {"type": "end"}
    kinds = [line.get("kind") for line in lines if line["type"] == "event"]
    assert kinds == ["stage", "mark"]
    events = [line for line in lines if line["type"] == "event"]
    assert all(0 <= e["t"] < 5 for e in events)
    assert meter.summary()["peak_footprint"] == 9


@macos
def test_footprint_grows_after_an_allocation():
    from stageload.meter import process_footprint

    before = process_footprint()
    block = bytearray(64 * 2**20)
    after = process_footprint()
    assert after - before > 32 * 2**20
    del block


@macos
def test_system_counters_are_in_range():
    from stageload.meter import memorystatus_level, swap_used

    assert swap_used() >= 0
    assert 0 <= memorystatus_level() <= 100


def test_a_failing_reader_is_noted_once_and_sampling_goes_on(tmp_path):
    path = tmp_path / "trace.jsonl"
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if 2 <= calls["n"] <= 6:
            raise OSError("sysctl failed")
        return calls["n"]

    readers = Readers(footprint=flaky, swap_used=lambda: 5, level=lambda: 77)
    deadline = time.monotonic() + 5
    with MemoryMeter(path, interval=0.01, readers=readers):
        while calls["n"] < 10 and time.monotonic() < deadline:
            time.sleep(0.01)
    assert calls["n"] >= 10, "the meter stopped sampling after the first failed read"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    errors = [r for r in records if r["type"] == "error"]
    assert [e["error"] for e in errors] == ["OSError: sysctl failed"]
    assert sum(r["type"] == "sample" for r in records) >= 3
    assert records[-1] == {"type": "end"}


def test_an_error_that_comes_back_after_a_recovery_is_noted_again(tmp_path):
    path = tmp_path / "trace.jsonl"
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] in (2, 3, 6):
            raise OSError("sysctl failed")
        return calls["n"]

    readers = Readers(footprint=flaky, swap_used=lambda: 5, level=lambda: 77)
    deadline = time.monotonic() + 5
    with MemoryMeter(path, interval=0.01, readers=readers):
        while calls["n"] < 8 and time.monotonic() < deadline:
            time.sleep(0.01)
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert sum(r["type"] == "error" for r in records) == 2
