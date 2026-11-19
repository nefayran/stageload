"""A lane's own start check: an executable gates/<lane> under the queue's home."""
from pathlib import Path

from stageload import queue


def gate(home: Path, lane: str, body: str) -> Path:
    path = home / "gates" / lane
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return path


def lane(home: Path, name: str) -> Path:
    path = home / "queue" / name
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_a_gate_that_says_no_gives_its_first_line(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_gate_seen", {})
    gate(tmp_path, "far", 'echo "the machine is hot: 87 C"; echo second; exit 1')
    assert queue.lane_gate(lane(tmp_path, "far")) == "the machine is hot: 87 C"


def test_no_gate_or_a_gate_that_says_yes_lets_the_run_start(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_gate_seen", {})
    assert queue.lane_gate(lane(tmp_path, "near")) == ""
    gate(tmp_path, "far", "exit 0")
    assert queue.lane_gate(lane(tmp_path, "far")) == ""


def test_a_gate_that_is_not_executable_is_left_alone(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_gate_seen", {})
    gate(tmp_path, "far", "exit 1").chmod(0o644)
    assert queue.lane_gate(lane(tmp_path, "far")) == ""


def test_the_answer_is_kept_for_a_while(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_gate_seen", {})
    path = gate(tmp_path, "far", 'echo hot; exit 1')
    far = lane(tmp_path, "far")
    assert queue.lane_gate(far) == "hot"
    path.write_text("#!/bin/sh\nexit 0\n")
    assert queue.lane_gate(far) == "hot"            # within 30 s the last answer stands
    monkeypatch.setattr(queue, "GATE_TTL_S", 0.0)
    assert queue.lane_gate(far) == ""


def test_the_gate_is_told_which_run_wants_to_start(tmp_path, monkeypatch):
    # a gate must know which ticket asks, so it can hold that one ticket and let the rest of the lane go
    monkeypatch.setattr(queue, "_gate_seen", {})
    gate(tmp_path, "ram", 'echo "$STAGELOAD_GATE_WHO $STAGELOAD_GATE_PID $(basename "$STAGELOAD_GATE_TICKET")"; exit 1')
    ram = lane(tmp_path, "ram")
    ticket = ram / "123-4567.json"
    assert queue.lane_gate(ram, ticket, {"who": "clip-5", "pid": 4567}) == "clip-5 4567 123-4567.json"


def test_each_ticket_has_its_own_answer(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_gate_seen", {})
    gate(tmp_path, "ram", '[ "$STAGELOAD_GATE_WHO" = held ] && { echo held by windows; exit 1; }; exit 0')
    ram = lane(tmp_path, "ram")
    assert queue.lane_gate(ram, ram / "1-1.json", {"who": "held", "pid": 1}) == "held by windows"
    assert queue.lane_gate(ram, ram / "2-2.json", {"who": "free", "pid": 2}) == ""


def test_a_gpu_clip_held_by_the_gate_lets_the_gpu_clips_behind_it_start(tmp_path, monkeypatch):
    # 14.11: Windows took the clip at the head of the Mac's ram lane. The gate held it, as it should, but
    # its ticket said "the GPU is taken", so no GPU clip behind it could start and the Mac stood idle with
    # 148 clips waiting until Windows finished. The held clip waits for Windows, not for the GPU or its place.
    ram = lane(tmp_path, "ram")
    head = ram / f"{100:020d}-100.json"
    data = {"state": "waiting", "who": "clip-1", "gpu": True}
    queue._write_json(head, data)
    queue._mark_gate_held(head, data, "Windows is rendering this clip", 1000.0)
    nxt = (ram / f"{200:020d}-200.json", {"state": "waiting", "who": "clip-2", "gpu": True})
    live = [(head, queue.json.loads(head.read_text())), nxt]
    monkeypatch.setattr(queue.time, "time", lambda: 1005.0)
    assert queue.lane_ahead(nxt[0], live, {}, None) == ([], "")
    # still so after the 20 minutes a waiting job may yield, while the gate keeps holding it
    later = 1000.0 + queue.BLOCKED_YIELD_S + 60
    live[0][1].update(blocked_at=later - 5)
    monkeypatch.setattr(queue.time, "time", lambda: later)
    assert queue.lane_ahead(nxt[0], live, {}, None) == ([], "")
