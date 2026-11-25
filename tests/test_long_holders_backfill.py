"""When every slot of a Mac lane is held by runs past LONG_HOLD_S, one short job may use the
express place.

25.11: drama-ep02-rough (expected 13 min) and w23-vo-pick (expected 2.1 h) held both light slots
for 4 to 5 hours. local-ci-686 (expected 2 minutes, AllKeep PRs #686 and #687 wait on it) waited
2.5 h, and NOW's showcase step timed out twice at its 0.5 h limit in the queue. The gate and the
budget still hold the short job, and a long job still waits for a slot.
"""

import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from stageload import queue
from stageload.queue import lane_ahead

GB = 2**30


@pytest.fixture
def machine(monkeypatch):
    for name in ("STAGELOAD_REMOTE_LANES", "STAGELOAD_SHARES", "STAGELOAD_LIGHT_SLOTS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(queue, "_ram", lambda: 48 * GB)


def _ago(minutes):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def _ticket(home, who, n, **data):
    lane_dir = home / "queue" / "light"
    lane_dir.mkdir(parents=True, exist_ok=True)
    path = lane_dir / f"{n:020d}-{os.getpid()}.json"
    entry = {"pid": os.getpid(), "who": who, "lane": "light", "cmd": "python job.py", **data}
    path.write_text(json.dumps(entry))
    return path, entry


def _lane(home, minutes=(285, 241)):
    return [
        _ticket(home, "drama-ep02-rough", 1, state="running", started=_ago(minutes[0]), expect_s=762),
        _ticket(home, "w23-vo-pick", 2, state="running", started=_ago(minutes[1]), expect_s=7721),
    ]


def test_a_short_job_has_its_turn_when_both_slots_are_held_for_hours(tmp_path, machine):
    live = _lane(tmp_path) + [_ticket(tmp_path, "local-ci-686", 3, state="waiting", expect_s=126)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2) == ([], "")


def test_a_long_job_still_waits_for_a_slot(tmp_path, machine):
    live = _lane(tmp_path) + [_ticket(tmp_path, "w16-bed", 3, state="waiting", expect_s=3600)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2)[1].startswith("2 ahead")


def test_no_place_is_lent_while_one_holder_is_young(tmp_path, machine):
    live = _lane(tmp_path, minutes=(285, 20)) + [
        _ticket(tmp_path, "local-ci-686", 3, state="waiting", expect_s=126)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2)[1].startswith("2 ahead")


def test_no_place_is_lent_while_a_slot_is_free(tmp_path, machine):
    live = _lane(tmp_path)[:1] + [_ticket(tmp_path, "local-ci-686", 3, state="waiting", expect_s=126)]
    assert lane_ahead(live[-1][0], live, {}, None, slots=2) == ([], "")
