import json

import pytest

from stageload.cli import main


def test_summary_prints_json(tmp_path, capsys):
    trace = tmp_path / "t.jsonl"
    trace.write_text(
        '{"type":"start","utc":"2026-10-04T10:00:00+00:00","interval":0.5}\n'
        '{"type":"sample","t":0.0,"footprint":5,"swap_used":1,"level":90}\n'
        '{"type":"end"}\n'
    )
    assert main(["summary", str(trace)]) == 0
    assert json.loads(capsys.readouterr().out)["peak_footprint"] == 5


def test_guard_needs_a_command(capsys):
    with pytest.raises(SystemExit) as exit_:
        main(["guard", "--"])
    assert exit_.value.code == 2
    assert "needs a command" in capsys.readouterr().err
