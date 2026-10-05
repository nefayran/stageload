"""The README's example runs as written."""

import re
from pathlib import Path

import pytest
import torch
from torch import nn

README = Path(__file__).parent.parent / "README.md"


def example(heading):
    text = README.read_text()
    section = text.split(f"## {heading}\n", 1)[1]
    return re.search(r"```python\n(.*?)```", section, re.S).group(1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 3)

    def forward(self, x):
        return self.lin(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(3, 2)

    def forward(self, x):
        return self.lin(x)


def test_the_pipeline_example_runs(tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.torch")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    monkeypatch.chdir(tmp_path)
    safetensors.save_file(Encoder().state_dict(), "encoder.safetensors")
    safetensors.save_file(Decoder().state_dict(), "decoder.safetensors")
    code = example("Use it in your own pipeline")
    assert 'DEVICE = "mps"' in code
    code = code.replace('DEVICE = "mps"', f'DEVICE = "{device}"')
    namespace = {"Encoder": Encoder, "Decoder": Decoder,
                 "image": torch.ones(1, 4, device=device)}
    exec(compile(code, "README.md", "exec"), namespace)
    assert namespace["mesh"].shape == (1, 2)
    assert namespace["models"].loaded() == []
