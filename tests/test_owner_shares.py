"""Owner shares (2026-10-26): within one deadline the group furthest below its share of the last 24 h goes next."""
import json
from datetime import datetime, timedelta, timezone

from stageload.queue import fair_order, load_shares, share_group


def _lane(tmp_path):
    lane = tmp_path / "home" / "queue" / "ram"
    lane.mkdir(parents=True)
    return lane


def _history(home, rows):
    now = datetime.now(timezone.utc)
    with open(home / "history.jsonl", "w") as f:
        for who, run_s, ago_s in rows:
            end = (now - timedelta(seconds=ago_s)).isoformat()
            f.write(json.dumps({"lane": "ram", "who": who, "run_s": run_s, "end": end}) + "\n")


def test_parse_and_groups(monkeypatch):
    monkeypatch.setenv("STAGELOAD_SHARES", "drama:drama=50,anomaly:wide|w[0-9]*|sh[0-9]*=40,box:omb|code=7,other=3")
    shares = load_shares(None)
    assert [g for g, _, _ in shares] == ["drama", "anomaly", "box", "other"]
    assert share_group("sh57", shares) == ("anomaly", 40.0)
    assert share_group("w10", shares) == ("anomaly", 40.0)
    assert share_group("wanderer", shares) == ("other", 3.0)
    assert share_group("drama", shares) == ("drama", 50.0)


def test_group_below_its_share_goes_first(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_SHARES", "drama:drama=50,anomaly:wide=40,other=10")
    lane = _lane(tmp_path)
    # in the last 24 h anomaly ran 10 h, drama 2 h: drama is far below its share and takes the next turns
    _history(lane.parent.parent, [("wide-w10-c1", 36000, 3600), ("drama-c1", 7200, 3600)])
    tickets = [(lane / f"{i:020d}-{i}.json", {"who": w, "state": "waiting"})
               for i, w in enumerate(["wide-a", "wide-b", "drama-a", "drama-b"], 1)]
    order = [p.name.split("-")[1] for p in fair_order(tickets, {})]
    assert order[0] == "3.json" and order[1] == "4.json"     # both drama jobs before anomaly's


def test_deadline_still_first(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_SHARES", "drama:drama=50,anomaly:wide|sh*=40")
    lane = _lane(tmp_path)
    _history(lane.parent.parent, [("sh58-c1", 36000, 3600)])
    tickets = [(lane / f"{1:020d}-1.json", {"who": "drama-a", "state": "waiting"}),
               (lane / f"{2:020d}-2.json", {"who": "sh58-c2", "state": "waiting", "deadline": "2026-10-28T08:30Z"})]
    order = [p.name for p in fair_order(tickets, {})]
    assert order[0] == f"{2:020d}-2.json"    # the short due first, though anomaly is over its share


def test_no_shares_keeps_turns(tmp_path, monkeypatch):
    monkeypatch.delenv("STAGELOAD_SHARES", raising=False)
    lane = _lane(tmp_path)
    tickets = [(lane / f"{i:020d}-{i}.json", {"who": w, "state": "waiting"})
               for i, w in enumerate(["a-1", "a-2", "b-1"], 1)]
    order = [p.name for p in fair_order(tickets, {})]
    assert order == [f"{1:020d}-1.json", f"{3:020d}-3.json", f"{2:020d}-2.json"]
