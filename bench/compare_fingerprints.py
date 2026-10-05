"""Compare the sampler fingerprints of two runs, call by call."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load(directory: Path) -> dict[str, dict[str, Any]]:
    records = json.loads((directory / "fingerprints.json").read_text())
    return {record["call"]: record for record in records}


def compare(a: Path, b: Path) -> list[dict[str, Any]]:
    """One row per call both runs made: equal hashes, or how far apart the tensors are."""
    ra, rb = _load(Path(a)), _load(Path(b))
    rows = []
    for call in [c for c in ra if c in rb]:
        ta = {t["name"]: t for t in ra[call]["tensors"]}
        tb = {t["name"]: t for t in rb[call]["tensors"]}
        for name in [n for n in ta if n in tb]:
            row: dict[str, Any] = {"call": call, "tensor": name,
                                   "equal": ta[name]["sha256"] == tb[name]["sha256"],
                                   "shape": ta[name]["shape"], "max_abs_diff": 0.0}
            if not row["equal"]:
                xa = torch.load(Path(a) / f"{call}.pt", weights_only=True)[name]
                xb = torch.load(Path(b) / f"{call}.pt", weights_only=True)[name]
                if xa.shape != xb.shape:
                    row["max_abs_diff"] = None
                    row["shape_b"] = list(xb.shape)
                else:
                    row["max_abs_diff"] = float((xa.double() - xb.double()).abs().max())
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.a, args.b), indent=2))


if __name__ == "__main__":
    main()
