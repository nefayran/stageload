"""Lane tiers before deadlines and shares: a run whose release slot is under 36 h away first, then
a short run that is not a video clip, then the rest in the usual order, where a floor group keeps
its part of the lane (here «thorn», 20 %). Without them one owner's long batch with an early
deadline kept every other owner waiting, releases due within hours included.
"""
import json
from datetime import datetime, timedelta, timezone

from stageload.queue import fair_order

SHARES = "drama:drama=60,anomaly:anomaly|sh[0-9]*=25,thorn:thorn=20,other=2"
DRAMA_FIRST = "1902-01-01T00:00Z"   # a priority deadline, as a cleaner writes it for a listed owner
H3_CMD = "zsh /x/.claude/skills/h3-video/scripts/h3_exec_convrot.sh /x/shot.json"


def _lane(tmp_path, monkeypatch):
    monkeypatch.setenv("STAGELOAD_SHARES", SHARES)
    monkeypatch.delenv("STAGELOAD_FLOORS", raising=False)
    lane = tmp_path / "home" / "queue" / "ram"
    lane.mkdir(parents=True)
    return lane


def _history(lane, rows):
    now = datetime.now(timezone.utc)
    with open(lane.parent.parent / "history.jsonl", "w") as f:
        for who, run_s, ago_s in rows:
            end = (now - timedelta(seconds=ago_s)).isoformat()
            row = {"lane": "ram", "who": who, "run_s": run_s, "end": end, "exit": 0}
            f.write(json.dumps(row) + "\n")


def _in(hours):
    return (datetime.now(timezone.utc) + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tickets(lane, rows):
    return [(lane / f"{i:020d}-{i}.json", dict(d, state="waiting")) for i, d in enumerate(rows, 1)]


def _whos(order, tickets):
    by = {p: d["who"] for p, d in tickets}
    return [by[p] for p in order]


def _drama_clips(n):
    return [{"who": f"drama-ep02-V09a{chr(97 + i)}", "cmd": H3_CMD, "deadline": DRAMA_FIRST,
             "expect_s": 734} for i in range(n)]


def test_a_clip_whose_slot_is_near_goes_before_the_priority_backlog(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    tickets = _tickets(lane, _drama_clips(3) + [
        {"who": "sh101-c5", "cmd": H3_CMD, "slot": _in(80), "expect_s": 832},   # 80 h away: waits
        {"who": "sh96-c1", "cmd": H3_CMD, "slot": _in(2), "expect_s": 700},
        {"who": "sh92-c5", "cmd": H3_CMD, "slot": _in(15), "expect_s": 700},
    ])
    whos = _whos(fair_order(tickets, {}), tickets)
    assert whos[:2] == ["sh96-c1", "sh92-c5"]          # nearest slot first
    assert whos.index("sh101-c5") > whos.index("drama-ep02-V09ac")


def test_short_runs_pass_the_h3_backlog(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    # this kind of run took 2-3 min; the ticket's own expect_s is only a guess
    _history(lane, [(f"anomaly-short-sh{n}-voices", 130 + n % 7, 90000) for n in range(80, 90)])
    tickets = _tickets(lane, _drama_clips(3) + [
        # a clip is never short, whatever its expect_s says
        {"who": "sh95-c10", "cmd": H3_CMD, "expect_s": 134, "deadline": "2000-07-15T09:32:44Z"},
        {"who": "anomaly-short-sh104-voices", "expect_s": 900, "deadline": "1904-01-01T00:00Z",
         "slot": _in(90)},
        {"who": "now-showcase-bubble_feast-render", "expect_s": 18},
        {"who": "anomaly_c-c07-voices", "expect_s": 2234},                      # long: not short
    ])
    whos = _whos(fair_order(tickets, {}), tickets)
    assert set(whos[:2]) == {"anomaly-short-sh104-voices", "now-showcase-bubble_feast-render"}
    assert whos[2] == "drama-ep02-V09aa"                # then drama first among the rest
    assert whos.index("sh95-c10") > whos.index("drama-ep02-V09ac")
    assert whos.index("anomaly_c-c07-voices") > whos.index("drama-ep02-V09ac")


def test_a_seconds_long_priority_run_goes_before_near_slots(tmp_path, monkeypatch):
    # a priority line asks for a 15-second narration first; the lane holds a dozen clips due
    # within hours and a priority run of two minutes, and the narration sat 29th for five hours
    lane = _lane(tmp_path, monkeypatch)
    _history(lane, [("narration-voice", 14 + n % 5, 90000) for n in range(10)]
             + [(f"anomaly-short-sh{n}-voices", 130 + n % 7, 90000) for n in range(80, 90)])
    tickets = _tickets(lane, [
        {"who": f"sh96-c{i}", "cmd": H3_CMD, "slot": _in(2 + i), "expect_s": 700} for i in range(12)
    ] + [
        {"who": "anomaly-short-sh104-voices", "deadline": "1904-01-01T00:00Z", "gpu": True},
        {"who": "narration-voice", "deadline": "1900-01-01T00:00Z", "gpu": True},
        {"who": "drama-ep02-V09aa", "cmd": H3_CMD, "deadline": DRAMA_FIRST, "expect_s": 734},
    ])
    whos = _whos(fair_order(tickets, {}), tickets)
    assert whos[0] == "narration-voice"
    assert whos[1] == "sh96-c0"                         # then the near slots, nearest first
    assert whos.index("anomaly-short-sh104-voices") > whos.index("sh96-c11")   # over 2 min
    assert whos.index("drama-ep02-V09aa") > whos.index("sh96-c11")      # a priority clip waits


def test_thorn_keeps_its_twenty_percent_behind_drama_first(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    thorn = [{"who": f"thorn-ep01-V0{i}b", "cmd": H3_CMD, "expect_s": 741} for i in range(10)]
    tickets = _tickets(lane, _drama_clips(20) + thorn)
    whos = _whos(fair_order(tickets, {}), tickets)
    assert whos[0].startswith("drama-")
    head = whos[:20]
    assert sum(w.startswith("thorn-") for w in head) == 4      # one clip in five, by run time
    assert whos.index("thorn-ep01-V00b") < whos.index("thorn-ep01-V01b")   # its own order kept


def test_a_day_without_thorn_does_not_give_it_the_lane_for_hours(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    _history(lane, [("drama-ep02-V01a", 3600, 3600 * h) for h in range(1, 24)])
    thorn = [{"who": f"thorn-ep01-V0{i}b", "cmd": H3_CMD, "expect_s": 741} for i in range(10)]
    tickets = _tickets(lane, _drama_clips(20) + thorn)
    whos = _whos(fair_order(tickets, {}), tickets)
    assert sum(w.startswith("thorn-") for w in whos[:10]) <= 2


def test_near_slot_and_short_still_go_before_thorn(tmp_path, monkeypatch):
    lane = _lane(tmp_path, monkeypatch)
    tickets = _tickets(lane, [{"who": "thorn-ep01-V04b", "cmd": H3_CMD, "expect_s": 741},
                              {"who": "now-showcase-x-render", "expect_s": 60},
                              {"who": "sh96-c1", "cmd": H3_CMD, "slot": _in(5)}])
    whos = _whos(fair_order(tickets, {}), tickets)
    assert whos == ["sh96-c1", "now-showcase-x-render", "thorn-ep01-V04b"]


def test_shares_report_names_the_floor(tmp_path, monkeypatch):
    from stageload.queue import share_report

    lane = _lane(tmp_path, monkeypatch)
    report = {r["group"]: r for r in share_report(lane.parent.parent)}
    assert report["thorn"]["floor"] == 20.0
    assert report["drama"]["floor"] is None


def test_an_h3_clip_with_a_tiny_guess_costs_what_h3_clips_really_take(tmp_path, monkeypatch):
    """A floor group's clips that guess 140 s, or whose kind's only finished runs found the clip already made (0.1 s),
    still cost what H3 clips take (~1400 s); counted at a tenth, the group took about twice its part of the lane."""
    lane = _lane(tmp_path, monkeypatch)
    with open(lane.parent.parent / "history.jsonl", "w") as f:
        for i in range(5):
            f.write(json.dumps({"lane": "ram", "who": f"wide-w20-c{i:03d}", "run_s": 1400, "exit": 0,
                                "cmd": H3_CMD}) + "\n")
        for i in range(10):     # a clip already made exits 0 in 0.1 s: not what a clip takes
            f.write(json.dumps({"lane": "ram", "who": f"thorn-ep01-V1{i}a{chr(97 + i)}", "run_s": 0.1, "exit": 0,
                                "cmd": H3_CMD}) + "\n")
    thorn = [{"who": f"thorn-ep01-V0{i}a{chr(97 + i)}", "cmd": H3_CMD, "expect_s": 140} for i in range(10)]
    tickets = _tickets(lane, _drama_clips(20) + thorn)
    whos = _whos(fair_order(tickets, {}), tickets)
    assert sum(w.startswith("thorn-") for w in whos[:20]) <= 4      # one clip in five, by real run time
