"""Find, import and check a Pixal3D-mac checkout."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

TESTED_COMMIT = "0be9e69a729432323dfb60a418801e5e9e7a1437"
MODULE_NAMES = (
    "parse_args",
    "load_runtime_deps",
    "_configure_fdg_environment",
    "resolve_device",
    "output_glb_path",
    "load_pipeline",
    "image_to_asset",
    "asset_to_glb",
    "save_mesh_checkpoint",
    "IMAGE_COND_CONFIGS",
    "build_image_cond_model",
)
PIPELINE_METHODS = (
    "from_pretrained",
    "run",
    "preprocess_image",
    "get_proj_cond_ss",
    "get_proj_cond_shape",
    "decode_latent",
)
EXTRACTOR_MODULE = "pixal3d.trainers.flow_matching.mixins.image_conditioned_proj"
DIAGNOSTIC_ENV = (
    "PIXAL3D_FP32_MODELS",
    "PIXAL3D_CPU_MODELS",
    "PIXAL3D_NAF_ANE_REPLACE",
    "PIXAL3D_NAF_ANE_WHOLE",
    "PIXAL3D_NAF_METAL",
    "PIXAL3D_DUMP_FIXTURES",
)


class PortError(RuntimeError):
    """The Pixal3D-mac checkout is missing or does not have what stageload needs."""


def _stderr(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


@dataclass
class Port:
    root: Path
    gm: ModuleType
    commit: str | None
    dirty: bool

    @property
    def pipeline_cls(self) -> Any:
        """Set by the port's ``load_runtime_deps()``; ``None`` before it runs."""
        return getattr(self.gm, "Pixal3DImageTo3DPipeline", None)

    @property
    def models_module(self) -> ModuleType:
        return importlib.import_module("pixal3d.models")

    @property
    def backbone_cls(self) -> Any:
        return importlib.import_module(EXTRACTOR_MODULE).DINOv3ViTModel


def find_port(explicit: str | os.PathLike[str] | None = None) -> Path:
    candidates = (
        [explicit]
        if explicit
        else [os.environ.get("PIXAL3D_MAC_DIR"), str(Path.home() / "local-llm" / "Pixal3D-mac")]
    )
    for candidate in candidates:
        if candidate and (Path(candidate).expanduser() / "generate_mps.py").is_file():
            return Path(candidate).expanduser().resolve()
    looked = ", ".join(str(c) for c in candidates if c)
    raise PortError(
        f"Pixal3D-mac not found (looked in: {looked}). Pass --port DIR or set PIXAL3D_MAC_DIR."
    )


def _git_state(root: Path) -> tuple[str | None, bool]:
    def git(*args: str) -> str:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        if result.returncode != 0:
            raise OSError(result.stderr.strip())
        return result.stdout.strip()

    try:
        commit = git("rev-parse", "HEAD")
        dirty = bool(git("status", "--porcelain", "--untracked-files=no"))
    except OSError:
        return None, False
    return commit, dirty


def load_port(root: Path, *, warn: Callable[[str], None] = _stderr) -> Port:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    gm = importlib.import_module("generate_mps")
    missing = [name for name in MODULE_NAMES if not hasattr(gm, name)]
    if missing:
        raise PortError(
            f"{root / 'generate_mps.py'} has no {', '.join(missing)}. stageload is tested with "
            f"Pixal3D-mac {TESTED_COMMIT[:7]}."
        )
    commit, dirty = _git_state(root)
    if commit != TESTED_COMMIT:
        at = commit[:7] if commit else "an unknown commit"
        warn(f"stageload: Pixal3D-mac is at {at}; stageload was tested with {TESTED_COMMIT[:7]}")
    if dirty:
        warn("stageload: the Pixal3D-mac checkout has local changes")
    return Port(root=root, gm=gm, commit=commit, dirty=dirty)


def check_pipeline_api(port: Port) -> None:
    cls = port.pipeline_cls
    if cls is None:
        raise PortError("call the port's load_runtime_deps() before building a pipeline")
    missing = [name for name in PIPELINE_METHODS if not hasattr(cls, name)]
    if missing:
        raise PortError(
            f"{cls.__name__} has no {', '.join(missing)}. stageload is tested with "
            f"Pixal3D-mac {TESTED_COMMIT[:7]}."
        )


def diagnostic_switches(environ: Mapping[str, str] = os.environ) -> list[str]:
    return [name for name in DIAGNOSTIC_ENV if environ.get(name, "").strip()]
