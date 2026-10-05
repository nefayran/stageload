"""Make identical loads return one shared instance for the duration of a block."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from typing import Any

_MISSING = object()


def _default_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Hashable:
    key = (args, tuple(sorted(kwargs.items())))
    try:
        hash(key)
    except TypeError:
        # two different objects can print the same (a large array's repr skips its middle),
        # so a call that cannot be keyed is not shared
        return _MISSING
    return key


@contextmanager
def share(
    owner: Any, attr: str, *, key: Callable[..., Hashable] | None = None
) -> Iterator[dict[Hashable, Any]]:
    """Within the block, calls to ``owner.attr`` with the same key return the first result.

    By default the key is the call's arguments; a call with an argument that cannot be hashed,
    such as a dict or a list, is passed through and not shared. ``key`` replaces that rule.

    ``owner`` can be a class (for a classmethod such as ``from_pretrained``) or a module. The
    original attribute is restored on exit; an attribute the owner inherited is removed again
    rather than copied onto it.
    """
    raw = vars(owner).get(attr, _MISSING)
    target = getattr(owner, attr)
    cache: dict[Hashable, Any] = {}

    def shared(*args: Any, **kwargs: Any) -> Any:
        cache_key = key(*args, **kwargs) if key is not None else _default_key(args, kwargs)
        if cache_key is _MISSING:
            return target(*args, **kwargs)
        if cache_key not in cache:
            cache[cache_key] = target(*args, **kwargs)
        return cache[cache_key]

    setattr(owner, attr, staticmethod(shared) if isinstance(owner, type) else shared)
    try:
        yield cache
    finally:
        if raw is _MISSING:
            delattr(owner, attr)
        else:
            setattr(owner, attr, raw)
