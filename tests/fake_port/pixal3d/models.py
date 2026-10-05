"""Fake checkpoint loader: every model is a tiny Linear that reports when it runs."""

import torch
from torch import nn

LOADED = []
ON_FORWARD = None


class FakeModel(nn.Module):
    def __init__(self, tag):
        super().__init__()
        self.tag = tag
        self.lin = nn.Linear(8, 8)
        self.in_channels = 8

    def forward(self, x=None, *args):
        if ON_FORWARD is not None:
            ON_FORWARD(self.tag)
        device = self.lin.weight.device
        if hasattr(x, "feats"):
            return type(x)(x.coords, self.lin(x.feats.to(device)))
        if isinstance(x, torch.Tensor) and x.shape[-1] == 8:
            return self.lin(x.to(device))
        return self.lin(torch.ones(1, 8, device=device))

    def upsample(self, slat, upsample_times=4):
        return self.forward(slat).coords


def from_pretrained(path, **kwargs):
    """Like a real checkpoint: the weights depend on the file, not on the random generator."""
    LOADED.append(path)
    model = FakeModel(path.rsplit("/", 1)[-1])
    generator = torch.Generator().manual_seed(sum(map(ord, model.tag)))
    with torch.no_grad():
        model.lin.weight.copy_(torch.randn(8, 8, generator=generator) * 0.3)
        model.lin.bias.copy_(torch.randn(8, generator=generator) * 0.1)
    return model
