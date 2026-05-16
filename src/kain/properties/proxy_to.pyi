from collections.abc import Callable
from typing import Any, TypeVar

__all__ = ("proxy_to",)

T = TypeVar("T")

def proxy_to(
    *mapping: Any,
    getter: Callable[[str], Callable[[Any], Any]] = ...,
    default: Any = ...,
    pre: Callable[[Any], Any] | None = None,
    safe: bool = True,
) -> Callable[[type[T]], type[T]]: ...
