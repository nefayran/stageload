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



def test_a_load_that_draws_from_python_and_numpy_leaves_them_where_they_were():
    np = pytest.importorskip("numpy")
    import random

    def noisy():
        random.random()
        np.random.rand()
        return nn.Linear(4, 4)

    models = StagedModels({"a": noisy}, stages={"one": ["a"]})
    random.seed(1)
    np.random.seed(1)
    models["a"]
    after = (random.random(), np.random.rand())
    random.seed(1)
    np.random.seed(1)
    assert after == (random.random(), np.random.rand())

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


def test_assigning_the_model_it_already_holds_keeps_it_working():
    models, _, _ = make("abc", STAGES)
    model = models["a"]
    models["a"] = model
    assert models["a"] is model
    model(torch.zeros(1, 4))


def test_a_replacement_that_shares_weights_keeps_them():
    models, _, _ = make("abc", STAGES)
    old = models["a"]
    new = nn.Linear(4, 4)
    new.weight = old.weight
    models["a"] = new
    assert new.weight.device.type == "cpu"
    assert old.bias.device.type == "meta"
    new(torch.zeros(1, 4))


def test_pop_hands_over_a_live_model_and_forgets_it():
    models, built, _ = make("abc", STAGES, pinned=["a"])
    loaded = models["a"]
    assert models.pop("a") is loaded
    loaded(torch.zeros(1, 4))
    popped = models.pop("b")
    assert built == ["a", "b"]
    popped(torch.zeros(1, 4))
    assert "a" not in models and "b" not in models
    assert models.pop("missing", None) is None
    with pytest.raises(KeyError):
        models.pop("missing")
    models.enter("three")


def test_popitem_takes_the_last_model():
    models, _, _ = make("ab", {"one": ["a", "b"]})
    name, model = models.popitem()
    assert name == "b"
    model(torch.zeros(1, 4))
    assert list(models) == ["a"]


def test_clear_releases_loaded_models_and_loads_nothing():
    models, built, sink = make("abc", STAGES)
    loaded = models["a"]
    models.clear()
    assert built == ["a"]
    assert len(models) == 0
    assert [e.kind for e in sink.events] == ["load", "release"]
    with pytest.raises(ReleasedModuleError):
        loaded(torch.zeros(1, 4))


def test_deleting_a_pinned_model_unpins_it():
    models, _, _ = make("abc", STAGES, pinned=["a"])
    models["a"]
    del models["a"]
    models["a"] = nn.Linear(4, 4)
    models.enter("two")
    assert "a" not in models.loaded()
