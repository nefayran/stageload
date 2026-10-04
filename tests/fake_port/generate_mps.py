"""Fake generate_mps.py with the entry points stageload uses."""

import argparse
from pathlib import Path

import torch

Pixal3DImageTo3DPipeline = None
Image = None
MODEL_PATH = "fake/Pixal3D"
IMAGE_COND_CONFIGS = {
    "ss": {"model_name": "fake/dinov3", "image_size": 512, "grid_resolution": 16},
    "shape_512": {"model_name": "fake/dinov3", "image_size": 512, "grid_resolution": 32,
                  "use_naf_upsample": True, "naf_target_size": 512},
    "shape_1024": {"model_name": "fake/dinov3", "image_size": 1024, "grid_resolution": 64,
                   "use_naf_upsample": True, "naf_target_size": 512},
    "tex_1024": {"model_name": "fake/dinov3", "image_size": 1024, "grid_resolution": 64,
                 "use_naf_upsample": True, "naf_target_size": 1024},
}
EXTRACTOR_KEYS = {
    "image_cond_model_ss": "ss",
    "image_cond_model_shape_512": "shape_512",
    "image_cond_model_shape_1024": "shape_1024",
    "image_cond_model_tex_1024": "tex_1024",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("image", nargs="?")
    parser.add_argument("--output", default="out.glb")
    parser.add_argument("--pipeline-type", default="1024_cascade")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--model-path", default=MODEL_PATH)
    parser.add_argument("--save-mesh", default=None)
    parser.add_argument("--low-vram", action="store_true")
    parser.add_argument("--tex-recalib", action="store_true")
    parser.add_argument("--tex-sat-boost", type=float, default=1.0)
    return parser.parse_args(argv)


def load_runtime_deps():
    global Pixal3DImageTo3DPipeline, Image
    from PIL import Image as _Image

    from pixal3d.pipelines import Pixal3DImageTo3DPipeline as _Pipeline

    Pixal3DImageTo3DPipeline = _Pipeline
    Image = _Image


def _configure_fdg_environment(args):
    return None


def resolve_device(name):
    return torch.device("cpu")


def output_glb_path(output):
    path = Path(output)
    return path if path.suffix.lower() == ".glb" else path.with_suffix(".glb")


def build_image_cond_model(config, device):
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        DinoV3ProjFeatureExtractor,
    )

    model = DinoV3ProjFeatureExtractor(**config)
    model.eval()
    model.to(device)
    return model


def load_pipeline(args, device=None):
    device = device or resolve_device(args.device)
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.to(device)
    for attr, key in EXTRACTOR_KEYS.items():
        setattr(pipeline, attr, build_image_cond_model(IMAGE_COND_CONFIGS[key], device))
        if getattr(pipeline, attr).use_naf_upsample:
            getattr(pipeline, attr)._load_naf()
    return pipeline


class Asset:
    def __init__(self, mesh):
        self.mesh = mesh
        self.vertices = [[0.0, 0.0, 0.0]]
        self.faces = [[0, 0, 0]]
        self.resolution = 1024


def image_to_asset(pipeline, image, args, device=None, *, tmp_dir=None):
    preprocessed = pipeline.preprocess_image(image)
    meshes, _ = pipeline.run(preprocessed, seed=args.seed, preprocess_image=False,
                             return_latent=True, pipeline_type=args.pipeline_type)
    return Asset(meshes[0])


def save_mesh_checkpoint(path, mesh, vertices, faces, resolution):
    Path(path).write_text("checkpoint\n")


def asset_to_glb(asset, glb_path, args):
    Path(glb_path).write_text(f"fake glb: {asset.mesh}\n")
