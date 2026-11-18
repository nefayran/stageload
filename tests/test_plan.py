"""`stageload plan` puts waiting runs in a chosen order and never touches a started one.

A planner (the board's director, 18.11.2026) places waiting work so that it keeps moving; the start
gates and the memory budget stay in force because a place only changes the order of waiting runs.
"""

import json

from stageload import order
from stageload.cli import main
from stageload.plan import place_deadline, plan
from stageload.queue import fair_order


def _ticket(home, lane, n, who, state="waiting", **extra):
    d = home / "queue" / lane
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{n:020d}-{40000 + n}.json"
    path.write_text(
        json.dumps({"who": who, "pid": 40000 + n, "state": state, "cmd": ["true"], **extra})
    )
    return path


def _alive(monkeypatch):
    from stageload import queue

    monkeypatch.setattr(queue, "_alive", lambda pid: True)


def test_first_puts_a_waiting_run_ahead_and_keeps_a_reason(tmp_path, monkeypatch):
    _alive(monkeypatch)
    a = _ticket(tmp_path, "ram", 1, "box-a")
    b = _ticket(tmp_path, "ram", 2, "anomaly-short-sh95-build")
    rows = plan(tmp_path, who="anomaly-*", place=0, by="director", why="its slot is the nearest")
    assert [r["done"] for r in rows] == ["place 0"]
    data = json.loads(b.read_text())
    assert (data["deadline"], data["planned_by"], data["planned_why"]) == (
        place_deadline(0),
        "director",
        "its slot is the nearest",
    )
    tickets = [(p, json.loads(p.read_text())) for p in (a, b)]
    assert fair_order(tickets, {})[0] == b


def test_a_started_run_is_never_touched(tmp_path, monkeypatch):
    _alive(monkeypatch)
    for state in ("running", "paused", "mem-wait", "gpu-wait"):
        _ticket(tmp_path, "ram", hash(state) % 1000, f"anomaly-{state}", state=state)
    before = {p: p.read_text() for p in (tmp_path / "queue" / "ram").iterdir()}
    rows = plan(tmp_path, owner="anomaly", place=0)
    assert {r["done"] for r in rows} == {"skipped"}
    assert {p: p.read_text() for p in before} == before


def test_h3_places_need_the_owners_word(tmp_path, monkeypatch):
    _alive(monkeypatch)
    clip = _ticket(tmp_path, "ram", 3, "sh95-c4", what="Short SH95: H3 video of shot 4 of 14")
    assert order.is_h3(json.loads(clip.read_text()))
    assert plan(tmp_path, who="sh95-*", place=0)[0]["done"] == "skipped"
    assert "deadline" not in json.loads(clip.read_text())
    assert plan(tmp_path, who="sh95-*", place=0, allow_h3=True)[0]["done"] == "place 0"


def test_prefer_lane_and_clear(tmp_path, monkeypatch):
    _alive(monkeypatch)
    clip = _ticket(tmp_path, "ram", 4, "sh95-c5", what="Short SH95: H3 video of shot 5 of 14")
    plan(tmp_path, who="sh95-c5", prefer_lane="win", by="director", why="Windows is free")
    data = json.loads(clip.read_text())
    assert (data["prefer_lane"], data["planned_why"]) == ("win", "Windows is free")
    assert "deadline" not in data  # a lane preference is not a place
    plan(tmp_path, who="sh95-c5", clear=True)
    assert not {"prefer_lane", "planned_by", "planned_why", "deadline"} & set(
        json.loads(clip.read_text())
    )


def test_dry_changes_nothing_and_the_cli_says_what_it_would_do(tmp_path, monkeypatch, capsys):
    _alive(monkeypatch)
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    t = _ticket(tmp_path, "light", 5, "anomaly-short-sh95-music")
    before = t.read_text()
    assert main(["plan", "--who", "anomaly-*", "--first", "--why", "slot near", "--dry"]) == 0
    assert "would place 0" in capsys.readouterr().out
    assert t.read_text() == before
    assert main(["plan", "--who", "nobody-*", "--first"]) == 1
    assert main(["plan", "--who", "anomaly-*"]) == 2


def test_the_waiting_run_keeps_the_plan_when_it_rewrites_its_entry(tmp_path, monkeypatch):
    _alive(monkeypatch)
    from stageload.queue import _write_json

    t = _ticket(tmp_path, "ram", 6, "anomaly-x")
    own = json.loads(t.read_text())
    plan(tmp_path, who="anomaly-x", place=3, prefer_lane="win", why="w")
    _write_json(t, {**own, "state": "waiting"})  # the run writes its own copy, made before the plan
    data = json.loads(t.read_text())
    assert (data["deadline"], data["prefer_lane"]) == (place_deadline(3), "win")
