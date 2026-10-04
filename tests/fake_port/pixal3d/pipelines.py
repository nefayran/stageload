"""Fake Pixal3D pipeline: the real method names, called in the real order."""

import torch
from torch import nn

from . import models

MODELS = {
    "sparse_structure_flow_model": "ckpts/ss_flow",
    "sparse_structure_decoder": "ckpts/ss_dec",
    "shape_slat_flow_model_512": "ckpts/flow_512",
    "shape_slat_flow_model_1024": "ckpts/flow_1024",
    "shape_slat_decoder": "ckpts/shape_dec",
    "tex_slat_flow_model_1024": "ckpts/tex_flow",
    "tex_slat_decoder": "ckpts/tex_dec",
}


class FakeBiRefNet:
    """Like the port's BiRefNet: a plain wrapper whose weights live in ``.model``."""

    def __init__(self):
        self.model = nn.Linear(8, 8)

    def to(self, device):
        self.model.to(device)
        return self

    def cpu(self):
        return self.to("cpu")

    def __call__(self, image):
        self.model(torch.ones(1, 8, device=self.model.weight.device))
        return image


class Pixal3DImageTo3DPipeline:
    def __init__(self, loaded=None):
        self.models = loaded or {}
        for model in self.models.values():
            model.eval()
        self.image_cond_model_ss = None
        self.image_cond_model_shape_512 = None
        self.image_cond_model_shape_1024 = None
        self.image_cond_model_tex_1024 = None
        self.rembg_model = None
        self.low_vram = True
        self._device = "cpu"

    @classmethod
    def from_pretrained(cls, path):
        loaded = {name: models.from_pretrained(f"{path}/{ckpt}") for name, ckpt in MODELS.items()}
        pipeline = cls(loaded)
        pipeline.rembg_model = FakeBiRefNet()
        return pipeline

    @property
    def device(self):
        return self._device

    def to(self, device):
        self._device = device
        if not self.low_vram:
            for model in self.models.values():
                model.to(device)
            self.rembg_model.to(device)
        return self

    def preprocess_image(self, image):
        return self.rembg_model(image)

    def get_proj_cond_ss(self, image, camera_angle_x=0.86, distance=2.0, mesh_scale=1.0):
        return self.image_cond_model_ss(image)

    def get_proj_cond_shape(self, image_cond_model, image, coords, **kwargs):
        return image_cond_model(image)

    def _use(self, name, call="forward", *args):
        model = self.models[name]
        if self.low_vram:
            model.to(self.device)
        out = getattr(model, call)(*args)
        if self.low_vram:
            model.cpu()
        return out

    def sample_sparse_structure(self, cond):
        self._use("sparse_structure_flow_model")
        self._use("sparse_structure_decoder")
        return "coords"

    def sample_shape_slat(self, cond, flow_model):
        if self.low_vram:
            flow_model.to(self.device)
        flow_model()
        if self.low_vram:
            flow_model.cpu()
        return "slat"

    def decode_latent(self, shape_slat, tex_slat, resolution):
        self._use("shape_slat_decoder")
        self._use("tex_slat_decoder")
        return ["mesh"]

    def run(self, image, seed=42, preprocess_image=True, return_latent=False,
            pipeline_type="1024_cascade", **kwargs):
        assert "shape_slat_flow_model_512" in self.models
        if preprocess_image:
            image = self.preprocess_image(image)
        cond_ss = self.get_proj_cond_ss(image)
        coords = self.sample_sparse_structure(cond_ss)
        cond_lr = self.get_proj_cond_shape(self.image_cond_model_shape_512, [image], coords)
        lr_slat = self.sample_shape_slat(cond_lr, self.models["shape_slat_flow_model_512"])
        hr_coords = self._use("shape_slat_decoder", "upsample", lr_slat)
        cond_hr = self.get_proj_cond_shape(self.image_cond_model_shape_1024, [image], hr_coords)
        flow_model_hr = self.models["shape_slat_flow_model_1024"]
        assert flow_model_hr.in_channels == 8
        shape_slat = self.sample_shape_slat(cond_hr, flow_model_hr)
        cond_tex = self.get_proj_cond_shape(self.image_cond_model_tex_1024, [image], shape_slat)
        tex_slat = self.sample_shape_slat(cond_tex, self.models["tex_slat_flow_model_1024"])
        meshes = self.decode_latent(shape_slat, tex_slat, 1024)
        return (meshes, (shape_slat, tex_slat, 1024)) if return_latent else meshes
