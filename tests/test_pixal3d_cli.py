import json
import sys

import pytest

from stageload.pixal3d.cli import main

macos = pytest.mark.skipif(sys.platform != "darwin", reason="the meter reads macOS counters")


def image_file(tmp_path):
    from PIL import Image

    path = tmp_path / "photo.png"
    Image.new("RGB", (8, 8), "white").save(path)
    return path


@macos
@pytest.mark.parametrize("mode", ["staged", "eager"])
def test_one_run_writes_a_glb_and_a_trace(fake_port_dir, tmp_path, mode, capsys):
    out = tmp_path / "out.glb"
    trace = tmp_path / "out.trace.jsonl"
    code = main([str(image_file(tmp_path)), "-o", str(out), "--port", str(fake_port_dir),
                 "--load", mode, "--trace", str(trace)])
    assert code == 0
    assert out.read_text().startswith("fake glb")
    text = trace.read_text()
    lines = [json.loads(line) for line in text.splitlines()]
    assert lines[0]["mode"] == mode and lines[0]["image"] == "photo.png"
    assert str(tmp_path) not in text
    stages = [line["stage"] for line in lines if line.get("kind") == "stage"]
    assert stages[0] == "setup" and stages[-1] == "export"
    assert "peak footprint" in capsys.readouterr().out


def test_options_after_the_image_are_not_swallowed(fake_port_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PIXAL3D_FP32_MODELS", "x")
    code = main(["x.png", "-o", str(tmp_path / "o.glb"), "--port", str(fake_port_dir)])
    assert code == 2
    assert "--load eager" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--flash-sdpa", "--load-mesh=m.npz", "--free-spent-models"])
def test_flags_for_other_branches_of_main_are_refused(fake_port_dir, tmp_path, flag, capsys):
    code = main(["x.png", "-o", str(tmp_path / "o.glb"), "--port", str(fake_port_dir),
                 "--load", "eager", "--", flag])
    assert code == 2
    assert "generate_mps.py" in capsys.readouterr().err


def test_a_missing_port_is_reported(tmp_path, capsys):
    code = main(["x.png", "-o", str(tmp_path / "o.glb"), "--port", str(tmp_path / "nowhere")])
    assert code == 2
    assert "Pixal3D-mac not found" in capsys.readouterr().err


@macos
def test_a_stopped_run_says_so_and_writes_no_glb(fake_port_dir, tmp_path, capsys):
    out = tmp_path / "out.glb"
    code = main([str(image_file(tmp_path)), "-o", str(out), "--port", str(fake_port_dir),
                 "--load", "eager", "--stop-at", "texture", "--fingerprint", str(tmp_path / "fp")])
    assert code == 0
    assert not out.exists()
    assert (tmp_path / "fp" / "fingerprints.json").exists()
    assert "stopped at texture as asked" in capsys.readouterr().out


def test_an_unknown_stop_stage_is_refused(fake_port_dir, tmp_path, capsys):
    code = main([str(image_file(tmp_path)), "-o", str(tmp_path / "o.glb"), "--port",
                 str(fake_port_dir), "--stop-at", "nowhere"])
    assert code == 2
    assert "--stop-at must be one of" in capsys.readouterr().err
