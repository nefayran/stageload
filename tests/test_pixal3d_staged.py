import importlib

import pytest

from stageload import ListSink
from stageload.pixal3d.plan import STAGES
from stageload.pixal3d.port import PortError
from stageload.pixal3d.staged import build_staged, generate
from stageload.release import is_released

TIMELINE = ["setup", "preprocess", "camera", "structure", "shape_512", "shape_1024", "texture",
            "decode", "export"]


def port_args(fake_port, tmp_path, *extra):
    return fake_port.gm.parse_args(["img.png", "--output", str(tmp_path / "out.glb"), *extra])


def extractor_module():
    return importlib.import_module("pixal3d.trainers.flow_matching.mixins.image_conditioned_proj")


def test_building_reads_no_model_and_shares_one_backbone(fake_port, fake_models, tmp_path):
    staged = build_staged(fake_port, port_args(fake_port, tmp_path), "cpu")
    assert fake_models.LOADED == []
    assert staged.models.loaded() == []
    assert set(staged.models) == {n for names in STAGES.values() for n in names}
    assert staged.pipeline.low_vram is False
    assert extractor_module().BUILT == {"backbone": 1, "naf": 1}
    assert len({id(m.model) for m in staged.extractors.values()}) == 1


def test_each_stage_holds_only_its_own_models(fake_port, fake_models, tmp_path):
    sink = ListSink()
    seen = []
    holder = {}

    def observe(tag):
        models = holder["staged"].models
        seen.append((tag, models.current_stage, sorted(models.loaded())))

    fake_models.ON_FORWARD = observe
    args = port_args(fake_port, tmp_path)
    holder["staged"] = build_staged(fake_port, args, "cpu", sink)
    fake_port.gm.image_to_asset(holder["staged"].pipeline, "image", args, "cpu")
    assert seen == [
        ("ss_flow", "structure", ["sparse_structure_flow_model"]),
        ("ss_dec", "structure", ["sparse_structure_decoder", "sparse_structure_flow_model"]),
        ("flow_512", "shape_512", ["shape_slat_flow_model_512"]),
        ("shape_dec", "shape_512", ["shape_slat_decoder", "shape_slat_flow_model_512"]),
        ("flow_1024", "shape_1024", ["shape_slat_flow_model_1024"]),
        ("tex_flow", "texture", ["tex_slat_flow_model_1024"]),
        ("shape_dec", "decode", ["shape_slat_decoder"]),
        ("tex_dec", "decode", ["shape_slat_decoder", "tex_slat_decoder"]),
    ]
    assert [e.stage for e in sink.events if e.kind == "stage"] == TIMELINE[:-1]
    assert [e.name for e in sink.events if e.kind == "reload"] == ["shape_slat_decoder"]


def test_rembg_and_the_extractors_are_released_when_no_longer_needed(fake_port, tmp_path):
    sink = ListSink()
    args = port_args(fake_port, tmp_path)
    staged = build_staged(fake_port, args, "cpu", sink)
    fake_port.gm.image_to_asset(staged.pipeline, "image", args, "cpu")
    assert is_released(staged.pipeline.rembg_model.model)
    assert all(is_released(m) for m in staged.extractors.values())
    released = [(e.name, e.stage) for e in sink.events if e.kind == "release"]
    assert ("rembg_model", "camera") in released
    assert ("image_cond_models", "decode") in released


def test_generate_writes_the_glb_and_frees_every_weight(fake_port, tmp_path):
    sink = ListSink()
    args = port_args(fake_port, tmp_path, "--save-mesh", str(tmp_path / "mesh.npz"))
    glb = generate(fake_port, args, "image", "cpu", mode="staged", sink=sink)
    assert glb.read_text().startswith("fake glb")
    assert (tmp_path / "mesh.npz").exists()
    assert [e.stage for e in sink.events if e.kind == "stage"] == TIMELINE


def test_eager_mode_runs_the_port_unchanged_and_only_marks_stages(fake_port, fake_models, tmp_path):
    sink = ListSink()
    generate(fake_port, port_args(fake_port, tmp_path), "image", "cpu", mode="eager", sink=sink)
    assert len(fake_models.LOADED) == 7
    assert [e.kind for e in sink.events] == ["stage"] * len(TIMELINE)
    assert [e.stage for e in sink.events] == TIMELINE


def test_an_unknown_mode_is_rejected(fake_port, tmp_path):
    with pytest.raises(ValueError, match="mode"):
        generate(fake_port, port_args(fake_port, tmp_path), "image", "cpu", mode="lazy")


def test_a_pipeline_without_a_planned_model_is_a_port_error(fake_port, tmp_path, monkeypatch):
    pipelines = importlib.import_module("pixal3d.pipelines")
    monkeypatch.delitem(pipelines.MODELS, "tex_slat_decoder")
    with pytest.raises(PortError, match="tex_slat_decoder"):
        build_staged(fake_port, port_args(fake_port, tmp_path), "cpu")
