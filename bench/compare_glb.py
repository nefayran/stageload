"""Compare two GLB files from Pixal3D: topology, vertex positions and the base color texture."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import trimesh


def _mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{path} holds no triangle mesh")
    return mesh


def _texture(mesh: trimesh.Trimesh) -> np.ndarray | None:
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "baseColorTexture", None)
    if image is None:
        image = getattr(material, "image", None)
    return None if image is None else np.asarray(image.convert("RGB"), dtype=np.int16)


def compare(a: Path, b: Path) -> dict[str, Any]:
    ma, mb = _mesh(Path(a)), _mesh(Path(b))
    same = ma.vertices.shape == mb.vertices.shape and np.array_equal(ma.faces, mb.faces)
    result: dict[str, Any] = {
        "vertices": {"a": int(len(ma.vertices)), "b": int(len(mb.vertices))},
        "faces": {"a": int(len(ma.faces)), "b": int(len(mb.faces))},
        "same_topology": bool(same),
        "vertex_max": None,
        "vertex_mean": None,
        "texture_max": None,
        "texture_psnr": None,
    }
    if same:
        distance = np.linalg.norm(ma.vertices - mb.vertices, axis=1)
        result["vertex_max"] = float(distance.max())
        result["vertex_mean"] = float(distance.mean())
    ta, tb = _texture(ma), _texture(mb)
    if ta is not None and tb is not None and ta.shape == tb.shape:
        diff = np.abs(ta - tb)
        mse = float((diff.astype(np.float64) ** 2).mean())
        result["texture_max"] = int(diff.max())
        result["texture_psnr"] = math.inf if mse == 0 else 10 * math.log10(255**2 / mse)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b), indent=2))


if __name__ == "__main__":
    main()
