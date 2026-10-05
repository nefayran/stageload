"""Build Pixal3D-mac's pipeline with staged loading, and run one image eager or staged."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
from torch import nn

from ..events import Event, EventSink
from ..registry import StagedModels
from ..release import release
from ..share import share
from .plan import EXTRACTORS, STAGES, install_stage_hooks
from .port import Port, PortError, check_pipeline_api
from .probe import Fingerprints, StopRun, stopping

MODES = ("staged", "eager")


class _Placeholder(nn.Module):
    """Stands in for a model while the port's constructor runs; remembers its checkpoint."""

    def __init__(self, path: str) -> None:
        super().__init__()
        self.stageload_path = path


@contextmanager
def _recording_loader(models_module: ModuleType) -> Iterator[Callable[..., nn.Module]]:
    original = models_module.from_pretrained

    def record(path: str, **kwargs: Any) -> nn.Module:
        if kwargs:
            raise TypeError(f"stageload only records checkpoint paths; got {sorted(kwargs)}")
        return _Placeholder(path)

    models_module.from_pretrained = record
    try:
        yield original
    finally:
        models_module.from_pretrained = original


def _model_loader(
    from_pretrained: Callable[..., nn.Module], path: str, device: Any
) -> Callable[[], nn.Module]:
    """Build the model directly on ``device``; if its constructor cannot, build it on the CPU.

    The port builds on the CPU and moves the model later. Doing the same here needs a second full
    copy during every load: on 05.10 it raised the per-stage peaks by 3 to 12 GB and the loading
    time from 24 s to 82 s. Building on MPS changes one thing the checkpoint does not cover: the
    sparse attention's rotary frequencies, a plain tensor computed in ``__init__``, come out a few
    bits different. Its effect stays inside the port's own run-to-run variation on MPS: two eager
    runs with the same seed differ by up to 1.77 in the 512 shape latent, a staged run and an
    eager run by 1.24.
    """

    def load() -> nn.Module:
        try:
            with torch.device(device):
                module = from_pretrained(path)
        except Exception as error:  # noqa: BLE001 - some constructors run CPU-only code
            warnings.warn(
                f"building {path} on {device} failed ({error!r}); building on the CPU instead",
                stacklevel=2,
            )
            module = from_pretrained(path)
        return module.to(device).eval()

    return load


def _modules_of(obj: Any) -> list[nn.Module]:
    """The module itself, or the modules a plain wrapper holds (the port's BiRefNet keeps its
    weights in ``.model``)."""
    if obj is None:
        return []
    if isinstance(obj, nn.Module):
        return [obj]
    return [value for value in vars(obj).values() if isinstance(value, nn.Module)]


def _build_extractors(port: Port, device: Any) -> dict[str, nn.Module]:
    configs = port.gm.IMAGE_COND_CONFIGS
    with share(port.backbone_cls, "from_pretrained"):
        extractors = {
            attr: port.gm.build_image_cond_model(configs[key], device)
            for attr, (key, _) in EXTRACTORS.items()
        }
    with_naf = [m for m in extractors.values() if getattr(m, "use_naf_upsample", False)]
    if with_naf:
        if not hasattr(with_naf[0], "_load_naf"):
            raise PortError("the DINOv3 extractor has no _load_naf(); the port has changed")
        with_naf[0]._load_naf()
        for module in with_naf[1:]:
            module.naf_model = with_naf[0].naf_model
    return extractors


@dataclass
class StagedPipeline:
    pipeline: Any
    models: StagedModels
    extractors: dict[str, nn.Module]
    sink: EventSink | None = None
    _rembg_released: bool = field(default=False, repr=False)
    _extractors_released: bool = field(default=False, repr=False)

    def enter_stage(self, stage: str) -> None:
        self.models.enter(stage)
        if stage == "camera":
            self._release_rembg()
        elif stage in ("decode", "export"):
            self._release_extractors()

    def close(self) -> None:
        self.models.close()
        self._release_extractors()
        self._release_rembg()

    def _release_rembg(self) -> None:
        if self._rembg_released:
            return
        self._rembg_released = True
        modules = _modules_of(getattr(self.pipeline, "rembg_model", None))
        if modules:
            stage = self.models.current_stage
            total = sum(release(m, name="rembg_model", stage=stage).nbytes for m in modules)
            self._emit("rembg_model", total)

    def _release_extractors(self) -> None:
        if self._extractors_released:
            return
        self._extractors_released = True
        modules = list(self.extractors.values())
        total = 0
        for i, module in enumerate(modules):
            stats = release(
                module,
                name="image_cond_models",
                stage=self.models.current_stage,
                empty_cache=i == len(modules) - 1,
            )
            total += stats.nbytes
        self._emit("image_cond_models", total)

    def _emit(self, name: str, nbytes: int) -> None:
        if self.sink is not None:
            event = Event("release", name=name, stage=self.models.current_stage, nbytes=nbytes)
            self.sink.emit(event)


def build_staged(
    port: Port,
    args: Any,
    device: Any,
    sink: EventSink | None = None,
    *,
    install_hooks: bool = True,
) -> StagedPipeline:
    """The port's own pipeline object with lazy models, shared extractors and stage hooks."""
    check_pipeline_api(port)
    with _recording_loader(port.models_module) as real_from_pretrained:
        pipeline = port.pipeline_cls.from_pretrained(args.model_path)
    paths = {
        name: getattr(module, "stageload_path", None) for name, module in pipeline.models.items()
    }
    planned = {name for names in STAGES.values() for name in names}
    missing = sorted(planned - {name for name, path in paths.items() if path})
    if missing:
        raise PortError(f"the pipeline has no {', '.join(missing)}; stageload's plan needs them")
    models = StagedModels(
        {name: _model_loader(real_from_pretrained, path, device) for name, path in paths.items()},
        stages=STAGES,
        events=sink,
    )
    models.enter("setup")
    pipeline.models = models
    pipeline.low_vram = False
    pipeline.to(device)
    extractors = _build_extractors(port, device)
    for attr, module in extractors.items():
        setattr(pipeline, attr, module)
    if getattr(args, "tex_recalib", False) or getattr(args, "tex_sat_boost", 1.0) != 1.0:
        port.gm._install_tex_pbr_recalib(
            pipeline,
            recalib_metallic=bool(getattr(args, "tex_recalib", False)),
            metallic_mean=float(getattr(args, "tex_metallic_mean", 0.0002)),
            metallic_std=float(getattr(args, "tex_metallic_std", 0.001)),
            sat_boost=float(getattr(args, "tex_sat_boost", 1.0)),
        )
    staged = StagedPipeline(pipeline, models, extractors, sink)
    if install_hooks:
        install_stage_hooks(pipeline, staged.enter_stage)
    return staged


def generate(
    port: Port,
    args: Any,
    image: Any,
    device: Any,
    *,
    mode: str,
    sink: EventSink | None = None,
    stop_at: str | None = None,
    fingerprint_dir: str | Path | None = None,
) -> Path | None:
    """One image to a GLB, the way the port's main() does it, with eager or staged loading.

    ``stop_at`` ends the run where that stage would begin and returns ``None``;
    ``fingerprint_dir`` receives a hash and a copy of every sampler's output.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
    if stop_at is not None and stop_at not in STAGES:
        raise ValueError(f"stop_at must be one of {list(STAGES)}, not {stop_at!r}")
    glb_path = Path(port.gm.output_glb_path(args.output))
    glb_path.parent.mkdir(parents=True, exist_ok=True)
    staged: StagedPipeline | None = None
    if mode == "staged":
        staged = build_staged(port, args, device, sink, install_hooks=False)
        pipeline, enter = staged.pipeline, staged.enter_stage
    else:

        def enter(stage: str) -> None:
            if sink is not None:
                sink.emit(Event("stage", stage=stage))

        enter("setup")
        pipeline = port.gm.load_pipeline(args, device)
    on_stage = stopping(enter, stop_at)
    install_stage_hooks(pipeline, on_stage)
    if fingerprint_dir is not None:
        Fingerprints(fingerprint_dir).install(pipeline)
    try:
        asset = port.gm.image_to_asset(pipeline, image, args, device, tmp_dir=glb_path.parent)
        if getattr(args, "save_mesh", None):
            port.gm.save_mesh_checkpoint(
                args.save_mesh, asset.mesh, asset.vertices, asset.faces, asset.resolution
            )
        on_stage("export")
    except StopRun as stop:
        if sink is not None:
            sink.emit(Event("mark", note=f"stopped at {stop.stage}"))
        if staged is not None:
            staged.close()
        return None
    port.gm.asset_to_glb(asset, glb_path, args)
    if staged is not None:
        staged.close()
    return glb_path
