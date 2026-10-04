"""Fake DINOv3 extractor with the attributes the adapter uses."""

import torch
from torch import nn

BUILT = {"backbone": 0, "naf": 0}


class DINOv3ViTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)

    @classmethod
    def from_pretrained(cls, name):
        BUILT["backbone"] += 1
        return cls()


class FakeNAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(8, 8)


class DinoV3ProjFeatureExtractor(nn.Module):
    def __init__(self, model_name, image_size, grid_resolution, use_naf_upsample=False,
                 naf_target_size=128):
        super().__init__()
        self.use_naf_upsample = use_naf_upsample
        self.model = DINOv3ViTModel.from_pretrained(model_name)
        self.naf_model = None
        self.register_buffer("grid", torch.zeros(grid_resolution))

    def _load_naf(self):
        if self.naf_model is None:
            BUILT["naf"] += 1
            self.naf_model = FakeNAF()

    def forward(self, image):
        return self.model.lin(torch.ones(1, 8, device=self.model.lin.weight.device))
