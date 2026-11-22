"""22.11.2026 Rodion: «у роликов есть строгое расписание, нужно сначала распределить
задачи так что бы успеть их сделать в срок».
The plans are copied from each channel's slots.json to ~/.stageload/slots.json."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

ITEM = re.compile(r"^(sh\d{2,3}|ch\d{2}c?|s\d{1,3}|ep\d{2}|e\d{1,3}|w\d{2}|r\d{2}|c\d{2})$")
PAST_S = 3600
_CACHE: dict[Path, tuple[tuple[int, int], dict]] = {}


def slots_path(home: Path) -> Path:
    """The copied slot plan, or an explicitly configured plan."""
    override = os.environ.get("STAGELOAD_SLOTS")
    return Path(override) if override else Path(home) / "slots.json"


def _time(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).timestamp()


def load(home: Path) -> dict:
    """Read slots as channel -> item -> UTC timestamps, reloading when the file changes."""
    path = slots_path(home)
    try:
        stat = path.stat()
    except OSError:
        _CACHE.pop(path, None)
        return {}
    key = (stat.st_mtime_ns, stat.st_size)
    cached = _CACHE.get(path)
    if cached and cached[0] == key:
        return cached[1]
    result = {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    if isinstance(data, dict):
        for channel, rows in data.items():
            if not isinstance(channel, str) or not isinstance(rows, list):
                continue
            items = {}
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item = row.get("item")
                if not isinstance(item, str) or not ITEM.fullmatch(item.lower()):
                    continue
                at = _time(row.get("at"))
                if at is not None:
                    items.setdefault(item.lower(), []).append(at)
            result[channel.lower()] = items
    _CACHE[path] = (key, result)
    return result


def slot_for(who: str, home: Path,
             now: datetime | str | float | None = None) -> str | None:
    """Find the first item in a run name and return its nearest still-relevant slot."""
    plans = load(home)
    if not plans:
        return None
    current = datetime.now(timezone.utc).timestamp() if now is None else _time(now)
    if current is None:
        return None
    tokens = [part for part in re.split(r"[-_.\s]+", str(who).lower()) if part]
    if tokens and tokens[0] == "win":
        tokens = tokens[1:]
    channel = tokens[0] if tokens and tokens[0] in plans else None
    for token in tokens:
        if not ITEM.fullmatch(token):
            continue
        if channel is not None:
            channels = [channel] if token in plans[channel] else []
        elif token.startswith("ep"):
            return None
        else:
            channels = [name for name, items in plans.items() if token in items]
        if len(channels) != 1:
            return None
        candidates = [stamp for stamp in plans[channels[0]][token] if stamp >= current - PAST_S]
        if not candidates:
            return None
        stamp = min(candidates)
        return datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return None
