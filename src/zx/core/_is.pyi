from collections.abc import Callable
from typing import Any

__all__ = (
    "Builtin",
    "Class",
    "Primitive",
    "awaitable",
    "builtin",
    "classOf",
    "collection",
    "coroutine",
    "function",
    "imported",
    "internal",
    "iterable",
    "mapping",
    "method",
    "module",
    "primitive",
    "subclass",
    "tty",
)

Builtin: Callable[..., bool]
Class: Callable[..., Any]
Primitive: Callable[..., bool]
tty: bool
awaitable: Callable[..., bool]
builtin: Callable[..., bool]
classOf: Callable[..., type[Any]]
collection: Callable[..., bool]
coroutine: Callable[..., bool]
function: Callable[..., bool]
imported: Callable[..., bool]
internal: Callable[..., bool]
iterable: Callable[..., bool]
mapping: Callable[..., bool]
method: Callable[..., bool]
module: Callable[..., bool]
primitive: Callable[..., bool]
subclass: Callable[..., bool]
