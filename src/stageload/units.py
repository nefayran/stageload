"""Sizes and durations as people type them on a command line."""

from __future__ import annotations

import re

_SIZE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([kmgt]?)(?:i?b)?\s*$", re.IGNORECASE)
_SIZE_FACTORS = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}
_DURATION = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(ms|s|m|h)?\s*$", re.IGNORECASE)
_SECONDS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}


def parse_size(text: str) -> int:
    """'8G' -> 8 GiB in bytes. Units are binary: K, M, G, T, with an optional 'B' or 'iB'."""
    match = _SIZE.match(text)
    if not match:
        raise ValueError(f"not a size: {text!r} (examples: 8G, 512M)")
    return int(float(match.group(1)) * _SIZE_FACTORS[match.group(2).lower()])


def parse_duration(text: str) -> float:
    """'30m' -> 1800.0 seconds. Units: ms, s, m, h; a bare number means seconds."""
    match = _DURATION.match(text)
    if not match:
        raise ValueError(f"not a duration: {text!r} (examples: 30m, 90s)")
    unit = (match.group(2) or "s").lower()
    return float(match.group(1)) * _SECONDS[unit]
