"""Two waiting jobs write disk-hold.json at the same moment (26.11, 02:3x UTC: W23 voices and an SH121 step failed
with FileNotFoundError '.tmp-disk-hold.json' -> 'disk-hold.json'). Each writer has its own temporary file."""
from __future__ import annotations

import json
import multiprocessing
from pathlib import Path

from stageload import queue


def _writer(path: str, n: int) -> None:
    for i in range(n):
        queue._write_json(Path(path), {"i": i})


def test_writers_in_parallel_never_lose_their_temp(tmp_path: Path) -> None:
    target = tmp_path / "disk-hold.json"
    ctx = multiprocessing.get_context("fork")
    procs = [ctx.Process(target=_writer, args=(str(target), 300)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    assert [p.exitcode for p in procs] == [0] * 6
    assert "i" in json.loads(target.read_text())
    assert not list(tmp_path.glob(".tmp-*"))
