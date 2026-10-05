import json

import pytest

np = pytest.importorskip("numpy")
trimesh = pytest.importorskip("trimesh")
pytest.importorskip("matplotlib")
Image = pytest.importorskip("PIL.Image")

from bench.compare_glb import compare  # noqa: E402
from bench.plot_trace import plot  # noqa: E402
from bench.summarize_runs import collect, table  # noqa: E402


def textured_box(path, shift=0.0, tint=255):
    mesh = trimesh.creation.box()
    mesh.vertices[0] += shift
    uv = np.random.default_rng(0).random((len(mesh.vertices), 2))
    texture = Image.new("RGB", (16, 16), (tint, 128, 64))
    mesh.visual = trimesh.visual.TextureVisuals(uv=uv, image=texture)
    mesh.export(path)
    return path


def test_identical_meshes_compare_equal(tmp_path):
    a = textured_box(tmp_path / "a.glb")
    b = textured_box(tmp_path / "b.glb")
    result = compare(a, b)
    assert result["same_topology"] is True
    assert result["vertex_max"] == 0.0
    assert result["texture_max"] == 0
    assert result["texture_psnr"] == float("inf")


def test_a_moved_vertex_and_a_tinted_texture_show_up(tmp_path):
    a = textured_box(tmp_path / "a.glb")
    b = textured_box(tmp_path / "b.glb", shift=0.25, tint=250)
    result = compare(a, b)
    assert result["vertex_max"] == pytest.approx(0.25 * 3**0.5, rel=1e-3)
    assert result["texture_max"] == 5


def write_trace(path, mode, footprints):
    lines = [{"type": "start", "utc": "2026-10-04T10:00:00+00:00", "interval": 0.5, "mode": mode}]
    lines.append({"type": "event", "kind": "stage", "stage": "structure", "t": 0.5})
    for i, footprint in enumerate(footprints):
        lines.append({"type": "sample", "t": float(i), "footprint": footprint * 2**30,
                      "swap_used": 2**30, "level": 70})
    lines.append({"type": "end"})
    path.write_text("".join(json.dumps(line) + "\n" for line in lines))
    return path


def test_plot_writes_svg_and_png(tmp_path):
    traces = [write_trace(tmp_path / "eager-1.trace.jsonl", "eager", [5, 20, 30, 10]),
              write_trace(tmp_path / "staged-1.trace.jsonl", "staged", [5, 9, 12, 6])]
    written = plot(traces, tmp_path / "memory")
    assert sorted(p.suffix for p in written) == [".png", ".svg"]
    assert all(p.stat().st_size > 1000 for p in written)


def test_collect_and_table_report_each_mode(tmp_path):
    write_trace(tmp_path / "eager-1.trace.jsonl", "eager", [5, 20, 30, 10])
    write_trace(tmp_path / "staged-1.trace.jsonl", "staged", [5, 9, 12, 6])
    (tmp_path / "runs.json").write_text(json.dumps([
        {"mode": "eager", "run": 1, "exit": 0}, {"mode": "staged", "run": 1, "exit": 0}]))
    summary = collect(tmp_path)
    assert summary["modes"]["eager"]["peak_footprint_gb"] == [30.0]
    assert summary["modes"]["staged"]["peak_footprint_gb"] == [12.0]
    text = table(summary)
    assert "| eager |" in text and "| staged |" in text

    from bench.summarize_runs import stage_table

    stages = stage_table(summary)
    assert "| stage | eager (GB) | staged (GB) |" in stages
    assert "| structure | 30.0 | 12.0 |" in stages


def write_fingerprints(directory, feats):
    import hashlib

    import torch

    directory.mkdir(parents=True)
    tensor = torch.tensor(feats, dtype=torch.float32)
    sha = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    record = {"call": "shape_slat_sampler#1",
              "tensors": [{"name": "feats", "shape": [len(feats)], "dtype": "float32",
                           "sha256": sha}]}
    (directory / "fingerprints.json").write_text(json.dumps([record]))
    torch.save({"feats": tensor}, directory / "shape_slat_sampler#1.pt")
    return directory


def test_fingerprints_report_equal_calls_and_the_size_of_a_difference(tmp_path):
    from bench.compare_fingerprints import compare as compare_fp

    a = write_fingerprints(tmp_path / "a", [1.0, 2.0, 3.0])
    b = write_fingerprints(tmp_path / "b", [1.0, 2.0, 3.0])
    c = write_fingerprints(tmp_path / "c", [1.0, 2.5, 3.0])
    assert compare_fp(a, b) == [{"call": "shape_slat_sampler#1", "tensor": "feats", "equal": True,
                                 "shape": [3], "max_abs_diff": 0.0}]
    assert compare_fp(a, c)[0]["max_abs_diff"] == 0.5
