"""Which models each stage of Pixal3D's 1024_cascade run needs, and where stages begin."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..hooks import on_call

STAGES: dict[str, tuple[str, ...]] = {
    "setup": (),
    "preprocess": (),
    "camera": (),
    "structure": ("sparse_structure_flow_model", "sparse_structure_decoder"),
    "shape_512": ("shape_slat_flow_model_512", "shape_slat_decoder"),
    "shape_1024": ("shape_slat_flow_model_1024",),
    "texture": ("tex_slat_flow_model_1024",),
    "decode": ("shape_slat_decoder", "tex_slat_decoder"),
    "export": (),
}

# the plan follows the 1024 cascade; the 1536 cascade runs other models in other stages
STAGED_PIPELINES = ("1024_cascade",)
# where a run can be stopped: setup has begun before any hook is installed
STOP_STAGES = tuple(stage for stage in STAGES if stage != "setup")

# pipeline attribute -> (IMAGE_COND_CONFIGS key, the stage that calls this extractor)
EXTRACTORS: dict[str, tuple[str, str]] = {
    "image_cond_model_ss": ("ss", "structure"),
    "image_cond_model_shape_512": ("shape_512", "shape_512"),
    "image_cond_model_shape_1024": ("shape_1024", "shape_1024"),
    "image_cond_model_tex_1024": ("tex_1024", "texture"),
}


def install_stage_hooks(pipeline: Any, on_stage: Callable[[str], None]) -> None:
    """Call ``on_stage(name)`` where each stage of the port's ``run()`` begins.

    ``preprocess_image`` opens ``preprocess`` and, when it returns, ``camera``;
    ``get_proj_cond_ss`` opens ``structure``; ``get_proj_cond_shape`` opens the stage of the
    extractor it is given; ``decode_latent`` opens ``decode``. The hooks stay installed: a
    pipeline is used for one run.
    """
    stage_of = {
        id(getattr(pipeline, attr)): stage
        for attr, (_, stage) in EXTRACTORS.items()
        if getattr(pipeline, attr, None) is not None
    }

    def cond_shape(*args: Any, **kwargs: Any) -> None:
        extractor = args[0] if args else kwargs.get("image_cond_model")
        stage = stage_of.get(id(extractor))
        if stage is not None:
            on_stage(stage)

    on_call(
        pipeline,
        "preprocess_image",
        before=lambda *a, **k: on_stage("preprocess"),
        after=lambda *a, **k: on_stage("camera"),
    )
    on_call(pipeline, "get_proj_cond_ss", before=lambda *a, **k: on_stage("structure"))
    on_call(pipeline, "get_proj_cond_shape", before=cond_shape)
    on_call(pipeline, "decode_latent", before=lambda *a, **k: on_stage("decode"))
