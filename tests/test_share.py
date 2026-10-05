import types

from stageload import share


class Backbone:
    built = 0

    def __init__(self, name):
        self.name = name

    @classmethod
    def from_pretrained(cls, name):
        cls.built += 1
        return cls(name)


class Child(Backbone):
    pass


def test_calls_with_the_same_arguments_share_one_instance():
    Backbone.built = 0
    with share(Backbone, "from_pretrained") as cache:
        a = Backbone.from_pretrained("dino")
        b = Backbone.from_pretrained("dino")
        c = Backbone.from_pretrained("other")
    assert a is b
    assert a is not c
    assert Backbone.built == 2
    assert len(cache) == 2


def test_the_original_attribute_comes_back_after_the_block():
    with share(Backbone, "from_pretrained"):
        pass
    assert isinstance(vars(Backbone)["from_pretrained"], classmethod)
    assert Backbone.from_pretrained("x") is not Backbone.from_pretrained("x")


def test_an_inherited_attribute_is_removed_again_not_copied():
    with share(Child, "from_pretrained"):
        assert isinstance(Child.from_pretrained("x"), Child)
    assert "from_pretrained" not in vars(Child)
    assert isinstance(Child.from_pretrained("y"), Child)


def test_a_custom_key_decides_which_calls_are_the_same():
    with share(Backbone, "from_pretrained", key=lambda name: name.lower()):
        assert Backbone.from_pretrained("DINO") is Backbone.from_pretrained("dino")


def test_module_functions_can_be_shared():
    module = types.ModuleType("loaders")
    module.load = lambda path: object()
    with share(module, "load"):
        assert module.load("p") is module.load("p")
    assert module.load("p") is not module.load("p")


def test_the_attribute_is_restored_when_the_block_raises():
    try:
        with share(Backbone, "from_pretrained"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert isinstance(vars(Backbone)["from_pretrained"], classmethod)


def test_calls_that_cannot_be_keyed_are_not_shared():
    Backbone.built = 0
    with share(Backbone, "from_pretrained") as cache:
        a = Backbone.from_pretrained({"layers": 12})
        b = Backbone.from_pretrained({"layers": 12})
    assert a is not b
    assert Backbone.built == 2
    assert cache == {}
