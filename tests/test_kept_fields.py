"""A waiting run rewrites its entry from memory; the fields a queue planner or a priority list wrote into the file stay."""
import json

from stageload.queue import KEPT_FIELDS, _write_json


def test_rewrite_keeps_what_others_wrote(tmp_path):
    entry = tmp_path / "0123-1.json"
    run_copy = {"who": "manga-ch08-mmhits", "state": "waiting"}
    _write_json(entry, dict(run_copy))
    on_disk = json.loads(entry.read_text())
    on_disk.update(deadline="2000-06-29T17:45:58Z", planned_by="planner", planned_why="waited 19 h", expect_s=286)
    entry.write_text(json.dumps(on_disk))
    run_copy["blocked"] = "the GPU"                       # the run's own rewrite, without the planner's fields
    _write_json(entry, run_copy)
    after = json.loads(entry.read_text())
    assert after["blocked"] == "the GPU"
    assert after["deadline"] == "2000-06-29T17:45:58Z" and after["planned_by"] == "planner" and after["expect_s"] == 286
    assert run_copy["deadline"] == "2000-06-29T17:45:58Z"   # the run's memory follows the file


def test_other_files_untouched(tmp_path):
    owners = tmp_path / "owners.json"
    owners.write_text(json.dumps({"deadline": "x"}))
    _write_json(owners, {"drama": 1})
    assert json.loads(owners.read_text()) == {"drama": 1}
    assert "deadline" in KEPT_FIELDS
