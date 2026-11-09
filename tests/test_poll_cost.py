import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from stageload import queue
from stageload.guard import GuardConfig, GuardIO, _Signals


def _lane(tmp_path):
    lane = tmp_path / "home" / "queue" / "ram"
    lane.mkdir(parents=True)
    return lane


def _count_reads(monkeypatch, path):
    reads = []
    read_text = Path.read_text

    def read(self, *args, **kwargs):
        if self == path:
            reads.append(self)
        return read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    return reads


def _bump_mtime(path):
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))


def test_history_starts_caches_lines_and_sees_appends(tmp_path, monkeypatch):
    lane = _lane(tmp_path)
    path = lane.parent.parent / "history.jsonl"
    end = "2026-10-27T12:00:00+00:00"
    row = {"lane": "ram", "who": "film-a", "end": end, "run_s": 60}
    path.write_text("\n".join([
        json.dumps({**row, "who": "outside-tail"}),
        *["bad line"] * 5000,
        json.dumps(row),
        json.dumps({**row, "lane": "codex", "who": "other"}),
        json.dumps({**row, "state": "waiting", "who": "waiting"}),
        json.dumps({"lane": "ram"}),
        "bad line",
    ]) + "\n")
    reads = _count_reads(monkeypatch, path)
    expected = {"film": int((datetime.fromisoformat(end).timestamp() - 60) * 1e9)}
    assert queue._history_starts(lane) == expected
    assert queue._history_starts(lane) == expected
    assert queue._history_starts(lane, limit=1) == {}
    assert len(queue._history_tail(path, 5000)) == 5000
    assert queue._history_tail(path, 1) == [None]
    assert len(reads) == 1

    with path.open("a") as file:
        file.write(json.dumps({**row, "who": "new-a", "run_s": 0}) + "\n")
    _bump_mtime(path)
    expected["new"] = int(datetime.fromisoformat(end).timestamp() * 1e9)
    assert queue._history_starts(lane) == expected
    assert len(reads) == 2


def test_share_use_preserves_window_and_typical_runs(tmp_path, monkeypatch):
    lane = _lane(tmp_path)
    now = datetime(2026, 10, 27, 12, tzinfo=timezone.utc).timestamp()

    def utc(seconds):
        return datetime.fromtimestamp(seconds, timezone.utc).isoformat()

    rows = [
        {"who": "film-a", "end": utc(now - 100), "run_s": 200},
        {"who": "film-b", "end": utc(now - 86400 + 50), "run_s": 100},
        {"who": "film-c", "end": utc(now - 90000), "run_s": 400},
        {"who": "wide-a", "end": utc(now + 50), "run_s": 100},
        {"who": "skip-a", "end": utc(now), "run_s": 900, "state": "waiting"},
        {"who": "skip-b", "end": utc(now), "run_s": 900, "lane": "codex"},
        {"who": "skip-c", "end": "invalid"},
    ]
    path = lane.parent.parent / "history.jsonl"
    path.write_text("\n".join(json.dumps({"lane": "ram", **row}) for row in rows)
                    + "\nbad line\n")
    reads = _count_reads(monkeypatch, path)
    tickets = [
        (lane / "running", {"who": "film-live", "state": "running", "started": utc(now - 30)}),
        (lane / "paused", {"who": "wide-paused", "state": "paused", "started": utc(now - 90)}),
    ]
    shares = [("film", ["film"], 50.0), ("shorts", ["wide"], 40.0)]
    calls = []
    share_group = queue.share_group

    def group(owner, shares):
        calls.append(owner)
        return share_group(owner, shares)

    monkeypatch.setattr(queue, "share_group", group)
    # Film: 200 + 50 inside the window + 30 running; wide: 50 before now.
    expected = {"film": 280.0, "shorts": 50.0,
                "_typical": {"film": 200.0, "wide": 100.0}}
    assert queue._share_use(tickets, shares, now) == expected
    assert calls == ["film", "wide"]
    calls.clear()
    assert queue._share_use(tickets, shares, now + 10)["film"] == 280.0
    assert calls == ["film", "wide"]
    assert len(reads) == 1


def test_live_tickets_caches_without_hiding_rewrites_or_dead_pids(tmp_path, monkeypatch):
    lane = _lane(tmp_path)
    path = lane / f"{1:020d}-123.json"
    row = {"pid": 123, "who": "film-a", "state": "waiting"}
    path.write_text(json.dumps(row))
    reads = _count_reads(monkeypatch, path)
    alive = [True]
    checks = []

    def is_alive(pid):
        checks.append(pid)
        return alive[0]

    monkeypatch.setattr(queue, "_alive", is_alive)
    assert queue._live_tickets(lane) == [(path, row)]
    queue._live_tickets(lane)[0][1]["who"] = "mutated"
    assert queue._live_tickets(lane) == [(path, row)]
    assert len(reads) == 1
    row["who"] = "film-b"
    path.write_text(json.dumps(row))
    _bump_mtime(path)
    assert queue._live_tickets(lane) == [(path, row)]
    assert len(reads) == 2
    alive[0] = False
    assert queue._live_tickets(lane) == []
    assert not path.exists()
    assert checks == [123] * 5
    assert queue._live_tickets(lane) == []
    assert path not in queue._TICKET_CACHE


@pytest.mark.parametrize("change", ["create", "signal", "remove", "none"])
def test_nap_until_change(tmp_path, change):
    lane = _lane(tmp_path)
    signals = _Signals()
    sleeps = []

    def sleep(seconds):
        sleeps.append(seconds)
        if change == "create":
            (lane / "new-ticket").write_text("")
            _bump_mtime(lane)
        elif change == "signal":
            signals.count = 1
        elif change == "remove":
            lane.rmdir()

    io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [], sleep=sleep)
    queue._nap_until_change(io, 3.5, signals, lane)
    assert sleeps == ([1.0, 1.0, 1.0, 0.5] if change == "none" else [1.0])


@pytest.mark.parametrize("count,choices,blocked,far", [
    (10, 1, True, True),
    (9, 1, True, False),
    (10, 2, True, False),
    (10, 1, False, False),
])
def test_only_far_single_lane_waiters_nap_longer(tmp_path, monkeypatch,
                                               count, choices, blocked, far):
    lane = _lane(tmp_path)
    signals = _Signals()
    naps = []
    monkeypatch.setattr(queue, "_live_tickets", lambda lane: [])
    monkeypatch.setattr(queue, "_active_hold", lambda lane: None)
    monkeypatch.setattr(queue, "_last_starts", lambda lane: {})
    monkeypatch.setattr(queue, "lane_ahead",
                        lambda *args: ([{}] * count, "ahead" if blocked else ""))
    monkeypatch.setattr(queue, "_take_slot", lambda locks: None)

    def nap(io, seconds, signals, *args):
        naps.append((seconds, args))
        signals.count = 1

    monkeypatch.setattr(queue, "_nap", nap)
    monkeypatch.setattr(queue, "_nap_until_change", nap)
    io = GuardIO(level=lambda: 90, swap_used=lambda: 0, processes=lambda: [], log=lambda s: None)
    assert queue._wait_turn([(lane, lane / "ticket", [])] * choices,
                            GuardConfig(poll=5), io, signals) is None
    assert naps == ([(queue.FAR_POLL_S, (lane,))] if far else [(5, ())])
