from collections.abc import (
    Callable,
)
from inspect import (
    isawaitable,
    isbuiltin,
    isclass,
    iscoroutine,
    isfunction,
    ismethod,
    ismodule,
)
from typing import Any

from kain.internals import (
    class_of,
    is_collection,
    is_from_builtin,
    is_from_primitive,
    is_imported_module,
    is_interactive,
    is_internal,
    is_iterable,
    is_mapping,
    is_primitive,
    is_subclass,
)

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

Builtin: Callable[..., bool] = is_from_builtin
Class: Callable[..., Any] = isclass
Primitive: Callable[..., bool] = is_from_primitive
tty: bool = is_interactive()
awaitable: Callable[..., bool] = isawaitable
builtin: Callable[..., bool] = isbuiltin
classOf: Callable[..., type[Any]] = class_of  # noqa: N816
collection: Callable[..., bool] = is_collection
coroutine: Callable[..., bool] = iscoroutine
function: Callable[..., bool] = isfunction
imported: Callable[..., bool] = is_imported_module
internal: Callable[..., bool] = is_internal
iterable: Callable[..., bool] = is_iterable
mapping: Callable[..., bool] = is_mapping
method: Callable[..., bool] = ismethod
module: Callable[..., bool] = ismodule
primitive: Callable[..., bool] = is_primitive
subclass: Callable[..., bool] = is_subclass
