from __future__ import annotations

from typing import Any, final, overload, override

from kain.properties.cached import (
    cached_property,
    class_cached_property,
    class_parent_cached_property,
    mixed_cached_property,
    mixed_parent_cached_property,
    post_cached_property,
    post_parent_cached_property,
    pre_cached_property,
    pre_parent_cached_property,
)
from kain.properties.class_property import class_property, mixed_property
from kain.properties.primitives import (
    AttributeExceptionError,
    BaseProperty,
    ContextFaultError,
    Nothing,
    PropertyError,
    ReadOnlyError,
    bound_property,
)
from kain.properties.proxy_to import proxy_to

__all__ = (
    "AttributeExceptionError",
    "BaseProperty",
    "ContextFaultError",
    "PropertyError",
    "ReadOnlyError",
    "bound_property",
    "cached_property",
    "class_cached_property",
    "class_parent_cached_property",
    "class_property",
    "mixed_cached_property",
    "mixed_parent_cached_property",
    "mixed_property",
    "pin",
    "post_cached_property",
    "post_parent_cached_property",
    "pre_cached_property",
    "pre_parent_cached_property",
    "proxy_to",
)
T_co: Any

@final
class pin[T_co](bound_property[T_co]):
    native: Any
    cls: Any
    any: Any
    pre: Any
    post: Any
    @overload
    def __get__(
        self,
        node: None,
        klass: Any = ...,
    ) -> pin[T_co]: ...
    @overload
    def __get__(
        self,
        node: object,
        klass: Any = ...,
    ) -> T_co: ...
    @override  # type: ignore[misc]
    def __get__(
        self,
        node: object | None,
        klass: Any = Nothing,
    ) -> pin[T_co] | T_co: ...
