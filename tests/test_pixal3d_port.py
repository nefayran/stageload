import shutil

import pytest

from stageload.pixal3d.port import (
    TESTED_COMMIT,
    PortError,
    check_pipeline_api,
    diagnostic_switches,
    find_port,
    load_port,
)


def test_the_fake_port_loads_and_passes_the_api_check(fake_port):
    check_pipeline_api(fake_port)
    assert fake_port.pipeline_cls.__name__ == "Pixal3DImageTo3DPipeline"
    assert fake_port.backbone_cls.__name__ == "DINOv3ViTModel"


def test_a_port_without_a_needed_entry_point_names_the_tested_commit(fake_port_dir, tmp_path):
    broken = tmp_path / "port"
    shutil.copytree(fake_port_dir, broken)
    script = broken / "generate_mps.py"
    script.write_text(script.read_text().replace("def asset_to_glb", "def _gone"))
    with pytest.raises(PortError, match=TESTED_COMMIT[:7]) as error:
        load_port(broken, warn=lambda message: None)
    assert "asset_to_glb" in str(error.value)


def test_a_different_commit_only_warns(fake_port_dir):
    messages = []
    load_port(fake_port_dir, warn=messages.append)
    assert any("tested with" in m for m in messages)


def test_find_port_prefers_the_explicit_directory(fake_port_dir, monkeypatch):
    monkeypatch.setenv("PIXAL3D_MAC_DIR", "/nonexistent")
    assert find_port(fake_port_dir) == fake_port_dir.resolve()


def test_find_port_explains_where_it_looked(tmp_path, monkeypatch):
    monkeypatch.setenv("PIXAL3D_MAC_DIR", str(tmp_path / "missing"))
    monkeypatch.setenv("HOME", str(tmp_path))
    with pytest.raises(PortError, match="PIXAL3D_MAC_DIR"):
        find_port(None)


def test_diagnostic_switches_are_listed():
    environ = {"PIXAL3D_CPU_MODELS": "tex_slat_decoder", "PIXAL3D_FP32_MODELS": " ", "HOME": "/x"}
    assert diagnostic_switches(environ) == ["PIXAL3D_CPU_MODELS"]
