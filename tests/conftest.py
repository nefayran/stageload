import sys
from pathlib import Path

import pytest

FAKE_PORT = Path(__file__).parent / "fake_port"


def _forget_port_modules():
    for name in list(sys.modules):
        if name == "generate_mps" or name == "pixal3d" or name.startswith("pixal3d."):
            del sys.modules[name]


@pytest.fixture
def fake_port_dir():
    _forget_port_modules()
    saved = list(sys.path)
    yield FAKE_PORT
    sys.path[:] = saved
    _forget_port_modules()


@pytest.fixture
def fake_port(fake_port_dir):
    from stageload.pixal3d.port import load_port

    port = load_port(fake_port_dir, warn=lambda message: None)
    port.gm.load_runtime_deps()
    return port


@pytest.fixture
def fake_models(fake_port):
    return fake_port.models_module
