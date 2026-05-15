# ruff: noqa: ANN401, N401

from __future__ import annotations

from contextlib import suppress
from typing import Any, TypeVar, final, overload, override

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

T_co = TypeVar("T_co", covariant=True)


@final
class pin[T_co](bound_property[T_co]):  # noqa: N801

    native = cached_property
    cls = class_cached_property
    any = mixed_cached_property
    pre = pre_cached_property
    post = post_cached_property

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

    @override
    def __get__(
        self,
        node: object | None,
        klass: Any = Nothing,
    ) -> pin[T_co] | T_co:
        if node is None:
            raise ContextFaultError(self.header_with_context(klass))

        if not hasattr(node, "__dict__"):
            raise TypeError(
                f"{self.header_with_context(node)}, {node=} has no __dict__",
            )

        with suppress(KeyError):
            return node.__dict__[self.name]

        value = self.function(node)
        node.__dict__[self.name] = value
        return value
