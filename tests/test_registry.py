import pytest
import torch
from torch import nn

from stageload import ListSink, ReleasedModuleError, StagedModels

STAGES = {"one": ["a", "b"], "two": ["b", "c"], "three": ["c"]}


def make(names, stages, **kwargs):
    built = []

    def loader(name):
        def build():
            built.append(name)
            return nn.Linear(4, 4)

        return build

    sink = ListSink()
    models = StagedModels({n: loader(n) for n in names}, stages=stages, events=sink, **kwargs)
    return models, built, sink


def test_nothing_loads_until_a_model_is_asked_for():
    models, built, _ = make("abc", STAGES)
    assert "a" in models
    assert len(models) == 3
    assert list(models) == ["a", "b", "c"]
    assert list(models.values()) == []
    assert models.loaded() == []
    assert built == []


def test_first_access_loads_once_and_reports_size():
    models, built, sink = make("abc", STAGES)
    first = models["a"]
    assert models["a"] is first
    assert built == ["a"]
    assert [e.kind for e in sink.events] == ["load"]
    assert sink.events[0].nbytes == (4 * 4 + 4) * 4


def test_entering_a_stage_releases_what_it_does_not_need():
    models, _, sink = make("abc", STAGES)
    models.enter("one")
    a = models["a"]
    models["b"]
    models.enter("two")
    assert models.loaded() == ["b"]
    assert a.weight.device.type == "meta"
    summary = [(e.kind, e.name, e.stage) for e in sink.events if e.kind in ("stage", "release")]
    assert summary == [("stage", None, "one"), ("stage", None, "two"), ("release", "a", "two")]


def test_pinned_models_survive_stage_changes_until_unpinned():
    models, _, _ = make("abc", STAGES, pinned=["a"])
    models.enter("one")
    models["a"]
    models.enter("three")
    assert models.loaded() == ["a"]
    models.unpin("a")
    models.enter("two")
    assert models.loaded() == []


def test_a_released_model_is_loaded_again_on_access():
    models, built, sink = make("abc", STAGES)
    models.enter("one")
    models["a"]
    models.enter("three")
    models.enter("one")
    models["a"]
    assert built == ["a", "a"]
    assert [e.kind for e in sink.events if e.name == "a"] == ["load", "release", "reload"]


def test_access_outside_the_current_stage_loads_and_is_flagged():
    models, _, sink = make("abc", STAGES)
    models.enter("three")
    models["a"]
    load = next(e for e in sink.events if e.kind == "load")
    assert (load.name, load.stage, load.note) == ("a", "three", "outside_stage")


def test_close_releases_everything_including_pinned_models():
    models, _, _ = make("abc", STAGES, pinned=["a"])
    models["a"]
    models["b"]
    models.close()
    assert models.loaded() == []
    assert models.current_stage is None


def test_a_module_registered_without_a_loader_cannot_come_back():
    models, _, _ = make("ab", {"one": ["a"], "two": ["b"]})
    models["extra"] = nn.Linear(2, 2)
    assert models.loaded() == ["extra"]
    models.close()
    with pytest.raises(ReleasedModuleError):
        models["extra"]


def test_unknown_names_raise_key_error_like_a_dict():
    models, _, _ = make("a", {"one": ["a"]})
    with pytest.raises(KeyError):
        models["zzz"]


def test_stages_must_name_registered_models():
    with pytest.raises(KeyError, match="no loader"):
        StagedModels({"a": lambda: nn.Linear(1, 1)}, stages={"one": ["a", "zzz"]})


def test_unknown_stage_is_an_error():
    models, _, _ = make("a", {"one": ["a"]})
    with pytest.raises(KeyError, match="unknown stage"):
        models.enter("nope")


def test_loading_a_model_does_not_change_the_random_numbers_that_follow():
    torch.manual_seed(0)
    expected = [torch.randn(3), torch.randn(3)]
    models = StagedModels({"a": lambda: nn.Linear(64, 64)}, stages={"one": ["a"]})
    torch.manual_seed(0)
    first = torch.randn(3)
    models["a"]
    second = torch.randn(3)
    assert torch.equal(first, expected[0])
    assert torch.equal(second, expected[1])


def test_random_state_can_be_left_alone_on_request():
    models = StagedModels(
        {"a": lambda: nn.Linear(64, 64)}, stages={"one": ["a"]}, preserve_rng=False
    )
    torch.manual_seed(0)
    torch.randn(3)
    models["a"]
    after_load = torch.randn(3)
    torch.manual_seed(0)
    torch.randn(3)
    assert not torch.equal(after_load, torch.randn(3))
