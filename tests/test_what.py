"""Every run can say in plain English what it makes and what for: `--what`.

2026-10-24: a queue page that showed only the --who labels did not say what each job was doing.
"""

import json
import os
import re
import sys

from stageload.cli import _parser, main
from stageload.guard import GuardConfig, GuardIO
from stageload.queue import list_queue, run_queued, run_spec, write_spec

WHAT = "Mixes an episode's sound: foley, music and voices, for its release"
TRUE = [sys.executable, "-c", "pass"]
# the line as the command center (:4420) and the HQ office (:4430) parse it
LINE = re.compile(r"^(\S+)\s+(\d+)\s+(running|waiting|gpu-wait|mem-wait|paused)\s+(\S+)"
                  r"(?:\s+\[gpu\])?\s+pid (\d+)\s+since (\S+)(.*)$")


def _io(home, rows):
    def swap():
        rows.extend(list_queue(home))
        return 0

    return GuardIO(level=lambda: 90, swap_used=swap, processes=lambda: [],
                   log=lambda line: None, compressor=lambda: 0)


def _ticket(home, data, lane="ram"):
    lane_dir = home / "queue" / lane
    lane_dir.mkdir(parents=True, exist_ok=True)
    base = {"pid": os.getpid(), "lane": lane, "cmd": "zsh audio/run_ram_audio.sh",
            "state": "running", "started": "2026-10-24T03:31:26+00:00"}
    (lane_dir / f"{1:020d}-{os.getpid()}.json").write_text(json.dumps({**base, **data}))


def test_the_flag_and_its_environment_default(monkeypatch):
    monkeypatch.delenv("STAGELOAD_WHAT", raising=False)
    assert _parser().parse_args(["run", "--", "true"]).what == ""
    assert _parser().parse_args(["run", "--what", WHAT, "--", "true"]).what == WHAT
    monkeypatch.setenv("STAGELOAD_WHAT", WHAT)
    assert _parser().parse_args(["run", "--", "true"]).what == WHAT


def test_the_ticket_and_the_history_keep_it(tmp_path):
    rows = []
    assert run_queued(TRUE, GuardConfig(poll=0.05), io=_io(tmp_path, rows), home=tmp_path,
                      who="w08-post-audio", what=WHAT, trace=None) == 0
    assert rows and all(row["what"] == WHAT for row in rows)
    record = json.loads((tmp_path / "history.jsonl").read_text().splitlines()[-1])
    assert record["what"] == WHAT


def test_a_spec_carries_it_and_an_old_spec_runs_without_it(tmp_path):
    spec = write_spec(TRUE, GuardConfig(poll=0.05), home=tmp_path, who="x", what=WHAT)
    assert json.loads(spec.read_text())["what"] == WHAT
    old = write_spec(TRUE, GuardConfig(poll=0.05), home=tmp_path, who="y")
    data = json.loads(old.read_text())
    del data["what"]
    old.write_text(json.dumps(data))
    assert run_spec(spec) == 0 and run_spec(old) == 0
    records = [json.loads(line) for line in (tmp_path / "history.jsonl").read_text().splitlines()]
    assert [r["what"] for r in records] == [WHAT, ""]


def test_the_queue_prints_it_and_the_readers_still_match(tmp_path, monkeypatch, capsys):
    _ticket(tmp_path, {"who": "w08-post-audio", "what": WHAT + "\n  twice  spaced",
                       "state": "mem-wait", "mem_wait": "budget: running 31.9 GB"})
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["queue"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    found = LINE.match(line)
    assert found and found.group(4) == "w08-post-audio" and found.group(5) == str(os.getpid())
    rest = found.group(7)
    assert re.search(r"memory: (.*?)\s{2}", rest).group(1) == "budget: running 31.9 GB"
    assert f"  what: {WHAT} twice spaced  zsh audio/run_ram_audio.sh" in rest
    assert main(["queue", "--json"]) == 0
    (row,) = json.loads(capsys.readouterr().out)
    assert row["what"] == WHAT + "\n  twice  spaced"


def test_an_old_ticket_without_it_prints_as_before(tmp_path, monkeypatch, capsys):
    _ticket(tmp_path, {"who": "w08-post-audio"})
    monkeypatch.setenv("STAGELOAD_HOME", str(tmp_path))
    assert main(["queue"]) == 0
    line = capsys.readouterr().out.splitlines()[0]
    assert "what:" not in line
    assert line.endswith("since 2026-10-24T03:31:26+00:00  zsh audio/run_ram_audio.sh")
    assert main(["queue", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["what"] == ""
