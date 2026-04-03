"""Pre-cached mixed property descriptors.

``pre_*`` descriptors are mixed properties that **only cache when accessed on
a class**.  When accessed on an instance, the value is computed every time
and is *not* stored in the instance cache.

This is useful for "pre-processing" or configuration-like properties that
should be memoized at the class level (because they are expensive to compute
and identical for all instances) but still callable on instances when needed.
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
    "pre_cached_property",
    "pre_parent_cached_property",
)

T = TypeVar("T")
R = TypeVar("R")


class pre_parent_cached_property(mixed_parent_cached_property[T, R]):
    """Mixed parent-cached descriptor that skips instance caching.

    Overrides ``__set__`` so that when ``node`` is an instance, the value
    is returned immediately without being written to the cache dict.  When
    ``node`` is a class, the normal parent-aware caching logic is used.
    """

    @override
    def __set__(self, node: object, value: R) -> R:
        """Skip instance caching; store directly on the class."""
        self.get_node(node)
        if not Is.Class(node):
            # Instance access: return the fresh value, do not memoize.
            return value
        return super().__set__(node, value)


class pre_cached_property(mixed_cached_property[T, R], Generic[T, R]):
    """Mixed cached descriptor that skips instance caching.

    Same semantics as ``pre_parent_cached_property`` but caches directly on
    the accessed class rather than the owning parent class.
    """

    @overload
    def __new__(
        cls,
        function: Callable[[T], R],
        **kw: object,
    ) -> pre_cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> pre_cached_property[Any, R]: ...
    def __new__(
        cls,
        *args: object,
        **kw: object,
    ) -> pre_cached_property[Any, R]:
        """Skip instance caching; return fresh values on instances."""
        return object.__new__(cls)

    @override
    def __set__(self, node: object, value: R) -> R:
        """Skip instance caching; return fresh values on instances."""
        self.get_node(node)
        if not Is.Class(node):
            return value
        return super().__set__(node, value)

    @class_cached_property
    def here(
        cls: type[pre_parent_cached_property[Any, Any]],
    ) -> type[pre_parent_cached_property[Any, Any]]:
        """Return the parent cached-property class this variant inherits from."""
        return pre_parent_cached_property
