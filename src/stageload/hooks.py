"""Run callbacks around one method of one object."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any


def on_call(
    obj: Any,
    method: str,
    *,
    before: Callable[..., None] | None = None,
    after: Callable[..., None] | None = None,
) -> Callable[[], None]:
    """Wrap ``obj.method`` on this instance only; return a function that removes the wrapper.

    ``before(*args, **kwargs)`` runs before the call, ``after(result, *args, **kwargs)`` after
    it. Wrappers stack; remove them in reverse order.
    """
    if before is None and after is None:
        raise ValueError("on_call needs a before or an after callback")
    original = getattr(obj, method)
    had_own = method in vars(obj)
    previous = vars(obj).get(method)

    @functools.wraps(original)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if before is not None:
            before(*args, **kwargs)
        result = original(*args, **kwargs)
        if after is not None:
            after(result, *args, **kwargs)
        return result

    setattr(obj, method, wrapper)

    def uninstall() -> None:
        if vars(obj).get(method) is not wrapper:
            raise RuntimeError(
                f"another wrapper was installed on {method!r} after this one; "
                "uninstall them in reverse order"
            )
        if had_own:
            setattr(obj, method, previous)
        else:
            delattr(obj, method)

    return uninstall
