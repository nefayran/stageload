import pytest

from stageload import on_call


class Pipe:
    def __init__(self):
        self.calls = []

    def cond(self, extractor, image):
        self.calls.append(extractor)
        return f"c-{extractor}"


def test_before_sees_the_arguments_and_runs_first():
    pipe = Pipe()
    seen = []

    def before(extractor, image):
        seen.append((extractor, list(pipe.calls)))

    on_call(pipe, "cond", before=before)
    assert pipe.cond("e512", "img") == "c-e512"
    assert seen == [("e512", [])]


def test_after_receives_the_result():
    pipe = Pipe()
    results = []
    on_call(pipe, "cond", after=lambda result, *a, **k: results.append(result))
    pipe.cond("x", None)
    assert results == ["c-x"]


def test_only_the_patched_instance_changes():
    pipe, other = Pipe(), Pipe()
    seen = []
    on_call(pipe, "cond", before=lambda *a, **k: seen.append(1))
    other.cond("x", None)
    assert seen == []


def test_uninstall_restores_the_class_method():
    pipe = Pipe()
    uninstall = on_call(pipe, "cond", before=lambda *a, **k: None)
    uninstall()
    assert "cond" not in vars(pipe)
    assert pipe.cond("x", None) == "c-x"


def test_uninstalling_out_of_order_is_refused():
    pipe = Pipe()
    first = on_call(pipe, "cond", before=lambda *a, **k: None)
    second = on_call(pipe, "cond", after=lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="reverse order"):
        first()
    second()
    first()
    assert "cond" not in vars(pipe)


def test_a_hook_needs_a_callback():
    with pytest.raises(ValueError):
        on_call(Pipe(), "cond")
