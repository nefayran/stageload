import sys
from pathlib import Path

import pytest

FAKE_PORT = Path(__file__).parent / "fake_port"


@pytest.fixture(autouse=True)
def _no_lane_hold(monkeypatch):
    """Runs of different owners follow each other in most tests: no hold between them, unless a
    test sets one."""
    monkeypatch.setenv("STAGELOAD_HOLD_S", "0")


# the absolute memory gate and pause, off: most tests fake swap of 10 GB or more, and the ones that
# run real commands read the real machine. The tests of the gate turn it back on
GATE_OFF = {"STAGELOAD_GATE_SWAP_GB": "100000", "STAGELOAD_GATE_COMPRESSOR_GB": "100000",
            "STAGELOAD_GATE_LEVEL": "0", "STAGELOAD_PAUSE_SWAP_GB": "100000",
            "STAGELOAD_PAUSE_COMPRESSOR_GB": "100000", "STAGELOAD_PAUSE_LEVEL": "0"}


@pytest.fixture(autouse=True)
def _one_ram_slot(monkeypatch):
    """Most tests are of one run at a time per lane; the tests of a second ram run set 2."""
    monkeypatch.setenv("STAGELOAD_RAM_SLOTS", "1")
    monkeypatch.setenv("STAGELOAD_LIGHT_SLOTS", "1")
    monkeypatch.setenv("STAGELOAD_WIN_SLOTS", "1")


@pytest.fixture(autouse=True)
def _no_memory_gate(monkeypatch):
    for name, value in GATE_OFF.items():
        monkeypatch.setenv(name, value)


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
