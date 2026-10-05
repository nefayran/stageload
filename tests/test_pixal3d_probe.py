import json

import pytest

from stageload import ListSink
from stageload.pixal3d.staged import generate

pytest.importorskip("numpy")

CALLS = ["sparse_structure_sampler#1", "shape_slat_sampler#1", "shape_slat_sampler#2",
         "tex_slat_sampler#1"]


def port_args(fake_port, tmp_path, name):
    return fake_port.gm.parse_args(["img.png", "--output", str(tmp_path / f"{name}.glb")])


def fingerprints(directory):
    records = json.loads((directory / "fingerprints.json").read_text())
    return [(r["call"], [(t["name"], t["sha256"]) for t in r["tensors"]]) for r in records]


def run(fake_port, tmp_path, mode, name, **kwargs):
    out = tmp_path / f"fp-{name}"
    result = generate(fake_port, port_args(fake_port, tmp_path, name), "image", "cpu",
                      mode=mode, fingerprint_dir=out, **kwargs)
    return result, out


def test_every_sampler_output_is_fingerprinted_with_its_tensors(fake_port, tmp_path):
    _, out = run(fake_port, tmp_path, "staged", "a")
    records = fingerprints(out)
    assert [call for call, _ in records] == CALLS
    assert [name for name, _ in records[1][1]] == ["coords", "feats"]
    assert all((out / f"{call}.pt").exists() for call in CALLS)


def test_staged_runs_produce_exactly_the_latents_of_eager_runs(fake_port, tmp_path):
    _, eager = run(fake_port, tmp_path, "eager", "eager")
    _, staged = run(fake_port, tmp_path, "staged", "staged")
    assert fingerprints(staged) == fingerprints(eager)


def test_without_preserved_random_state_staged_runs_would_drift(fake_port, tmp_path, monkeypatch):
    import contextlib

    import stageload.registry as registry

    _, eager = run(fake_port, tmp_path, "eager", "eager")
    monkeypatch.setattr(registry, "preserved_rng", contextlib.nullcontext)
    _, staged = run(fake_port, tmp_path, "staged", "staged")
    assert fingerprints(staged)[0] == fingerprints(eager)[0]
    assert fingerprints(staged) != fingerprints(eager)


@pytest.mark.parametrize("mode", ["eager", "staged"])
def test_a_run_can_stop_where_a_stage_begins(fake_port, tmp_path, mode):
    sink = ListSink()
    result, out = run(fake_port, tmp_path, mode, mode, stop_at="texture", sink=sink)
    assert result is None
    assert not (tmp_path / f"{mode}.glb").exists()
    assert [call for call, _ in fingerprints(out)] == CALLS[:3]
    stages = [e.stage for e in sink.events if e.kind == "stage"]
    assert stages[-1] == "shape_1024"
    assert any(e.kind == "mark" and e.note == "stopped at texture" for e in sink.events)
