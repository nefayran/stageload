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

    def forward(self, *args):
        if ON_FORWARD is not None:
            ON_FORWARD(self.tag)
        return self.lin(torch.ones(1, 8, device=self.lin.weight.device))

    def upsample(self, slat, upsample_times=4):
        return self.forward(slat)


def from_pretrained(path, **kwargs):
    LOADED.append(path)
    return FakeModel(path.rsplit("/", 1)[-1])
