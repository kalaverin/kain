"""Post-cached mixed property descriptors.

``post_*`` descriptors are mixed properties that **only cache when accessed on
an instance**.  When accessed on a class, the value is computed every time and
is *not* stored in the class cache.

This is useful for properties that are expensive per-instance but should still
be readable directly from the class (e.g. for introspection or default-value
inspection) without polluting the class-level cache.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar, overload, override

from kain.internals import Is
from kain.properties.cached.klass import class_cached_property
from kain.properties.cached.mixed import (
    mixed_cached_property,
    mixed_parent_cached_property,
)

__all__ = (
    "post_cached_property",
    "post_parent_cached_property",
)

T = TypeVar("T")
R = TypeVar("R")


class post_parent_cached_property(mixed_parent_cached_property[T, R]):
    """Mixed parent-cached descriptor that skips class caching.

    Overrides ``__set__`` so that when ``node`` is a class, the value is
    returned immediately without being written to the cache dict.  When
    ``node`` is an instance, normal instance caching applies.
    """

    @override
    def __set__(self, node: object, value: R) -> R:
        """Skip class caching; store directly on the instance."""
        self.get_node(node)
        if Is.Class(node):
            # Class access: return the fresh value, do not memoize.
            return value
        return super().__set__(node, value)


class post_cached_property(mixed_cached_property[T, R], Generic[T, R]):
    """Mixed cached descriptor that skips class caching.

    Same semantics as ``post_parent_cached_property`` but caches directly on
    the accessed instance without owner-resolution for the class side.
    """

    @overload
    def __new__(
        cls,
        function: Callable[[T], R],
        **kw: object,
    ) -> post_cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> post_cached_property[Any, R]: ...
    def __new__(
        cls,
        *args: object,
        **kw: object,
    ) -> post_cached_property[Any, R]:
        """Skip class caching; return fresh values on classes."""
        return object.__new__(cls)

    @override
    def __set__(self, node: object, value: R) -> R:
        """Skip class caching; return fresh values on classes."""
        self.get_node(node)
        if Is.Class(node):
            return value
        return super().__set__(node, value)

    @class_cached_property
    def here(
        cls: type[post_parent_cached_property[Any, Any]],
    ) -> type[post_parent_cached_property[Any, Any]]:
        """Return the parent cached-property class this variant inherits from."""
        return post_parent_cached_property
