"""A video's release slot is the deadline of the job that makes it."""

import json
import sys

import pytest

from stageload import queue, slots
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import fair_order, run_queued

PLAN = {
    "anomaly": [
        {"item": "SH113", "at": "2026-11-27T22:30:00Z"},
        {"item": "W21", "at": "2026-11-27T11:00:00Z"},
        {"item": "C07", "at": "2026-11-23T11:00:00Z"},
        {"item": "SH99", "at": "2026-11-20T22:30:00Z"},
    ],
    "drama": [{"item": "EP02", "at": "2026-11-30T23:00:00Z"}],
    "thorn": [{"item": "EP01", "at": "2026-11-29T23:00:00Z"}],
}
NOW = "2026-11-22T16:00:00Z"


def _slots_file(home, data=PLAN):
    (home / "slots.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.parametrize(
    ("who", "expected"),
    [
        ("sh113-c07", "2026-11-27T22:30:00Z"),
        ("sh199-c07", None),
        ("win-sh113-c3", "2026-11-27T22:30:00Z"),
        ("win-anomaly-c07-build", "2026-11-23T11:00:00Z"),
        ("anomaly-c07-build", "2026-11-23T11:00:00Z"),
        ("anomaly-sh199-c07", None),
        ("anomaly-short-sh113-build", "2026-11-27T22:30:00Z"),
        ("other-drama-sh113", "2026-11-27T22:30:00Z"),
        ("keys-find-w21_sukuna", "2026-11-27T11:00:00Z"),
        ("wide-w21-c002", "2026-11-27T11:00:00Z"),
        ("drama-ep02-v24ag", "2026-11-30T23:00:00Z"),
        ("thorn-ep01-v10i", "2026-11-29T23:00:00Z"),
        ("ep02-x", None),
        ("other-drama-ep02", None),
        ("sh99-c07", None),
        ("asr-kotoba", None),
        ("now-showcase-sheet_tears", None),
        ("", None),
    ],
)
def test_slot_for_picks_the_first_item_token_in_the_right_channel(tmp_path, monkeypatch, who, expected):
    monkeypatch.delenv("STAGELOAD_SLOTS", raising=False)
    _slots_file(tmp_path)
    assert slots.slot_for(who, tmp_path, NOW) == expected


def test_slot_for_skips_ambiguous_items(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGELOAD_SLOTS", raising=False)
    _slots_file(tmp_path, {**PLAN, "thorn": PLAN["thorn"] + [
        {"item": "EP02", "at": "2026-12-01T23:00:00Z"},
        {"item": "SH113", "at": "2026-12-01T23:00:00Z"},
    ]})
    assert slots.slot_for("ep02-x", tmp_path, NOW) is None
    assert slots.slot_for("drama-ep02-v1", tmp_path, NOW) == "2026-11-30T23:00:00Z"
    assert slots.slot_for("sh113-x", tmp_path, NOW) is None
    assert slots.slot_for("anomaly-sh113-x", tmp_path, NOW) == "2026-11-27T22:30:00Z"


def test_slot_for_does_not_match_an_episode_in_another_channel(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGELOAD_SLOTS", raising=False)
    _slots_file(tmp_path, {"thorn": [{"item": "EP02", "at": "2026-12-01T23:00:00Z"}]})
    assert slots.slot_for("drama-ep02-v1", tmp_path, NOW) is None


def test_slot_for_uses_the_earliest_recent_duplicate_row(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGELOAD_SLOTS", raising=False)
    _slots_file(tmp_path, {"anomaly": [
        {"item": "SH113", "at": "2026-11-23T11:00:00Z"},
        {"item": "SH113", "at": "2026-11-22T15:30:00Z"},
        {"item": "SH113", "at": "2026-11-22T14:00:00Z"},
    ]})
    assert slots.slot_for("sh113-c1", tmp_path, NOW) == "2026-11-22T15:30:00Z"


@pytest.mark.parametrize("contents", [None, "{invalid"])
def test_slot_for_missing_or_invalid_plan_returns_none(tmp_path, monkeypatch, contents):
    monkeypatch.delenv("STAGELOAD_SLOTS", raising=False)
    if contents is not None:
        (tmp_path / "slots.json").write_text(contents, encoding="utf-8")
    assert slots.slot_for("sh113-c1", tmp_path, NOW) is None


def test_slot_for_uses_the_environment_path(tmp_path, monkeypatch):
    override = tmp_path / "other.json"
    override.write_text(json.dumps({"other": [{"item": "SH113", "at": "2026-11-25T00:00:00Z"}]}),
                        encoding="utf-8")
    monkeypatch.setenv("STAGELOAD_SLOTS", str(override))
    assert slots.slot_for("sh113-c1", tmp_path, NOW) == "2026-11-25T00:00:00Z"


@pytest.fixture
def machine(monkeypatch):
    for name, value in {
        "STAGELOAD_GATE_SWAP_GB": "6",
        "STAGELOAD_GATE_COMPRESSOR_GB": "8",
        "STAGELOAD_GATE_LEVEL": "35",
        "STAGELOAD_RAM_SLOTS": "2",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("STAGELOAD_RESERVE_GB", raising=False)
    monkeypatch.delenv("STAGELOAD_REMOTE_LANES", raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * 2**30)


class _Readings:
    def __init__(self):
        self.logs = []

    def io(self):
        return GuardIO(level=lambda: 80, swap_used=lambda: 0, processes=lambda: [],
                       log=self.logs.append, compressor=lambda: 2**30)


def test_run_queued_uses_a_slot_as_deadline_only_when_no_deadline_is_given(
        tmp_path, monkeypatch, machine):
    slot = "2026-11-25T00:00:00Z"
    (tmp_path / "slots.json").write_text(
        json.dumps({"anomaly": [{"item": "SH113", "at": slot}]}), encoding="utf-8")
    captured = []
    monkeypatch.setattr(queue, "_write_queued", lambda home, entry: captured.append(dict(entry)))
    args = ([sys.executable, "-c", "pass"], GuardConfig(poll=0.05, grace=1.0, wait_free=0))

    assert run_queued(*args, io=_Readings().io(), home=tmp_path, who="sh113-c4",
                      trace=None) == 0
    assert captured[-1]["deadline"] == slot
    assert captured[-1]["deadline_by"] == "slot"
    assert captured[-1]["slot"] == slot

    assert run_queued(*args, io=_Readings().io(), home=tmp_path, who="sh113-c4",
                      deadline="2026-11-25T00:00:00Z", trace=None) == 0
    assert captured[-1]["deadline"] == "2026-11-25T00:00:00Z"
    assert "deadline_by" not in captured[-1]

    assert run_queued(*args, io=_Readings().io(), home=tmp_path, who="asr-kotoba",
                      trace=None) == 0
    assert "deadline" not in captured[-1]


def test_fair_order_uses_the_release_slot_deadline(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_NEAR_SLOT_H", "1")
    monkeypatch.delenv("STAGELOAD_SHARES", raising=False)
    lane = tmp_path / "queue" / "ram"
    lane.mkdir(parents=True)
    slot = "2026-11-27T16:00:00Z"
    tickets = [
        (lane / "1-1.json", {"who": "drama-ep02-v01", "cmd": "h3_exec_convrot.sh",
                             "expect_s": 3000, "state": "waiting"}),
        (lane / "2-2.json", {"who": "sh113-c1", "cmd": "h3_exec_convrot.sh",
                             "expect_s": 3000, "deadline": slot, "deadline_by": "slot",
                             "slot": slot, "state": "waiting"}),
    ]
    assert fair_order(tickets, {})[0] == tickets[1][0]
