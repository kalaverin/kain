"""Pre-cached mixed property descriptors.

``pre_*`` descriptors are mixed properties that **only cache when accessed on
a class**.  When accessed on an instance, the value is computed every time
and is *not* stored in the instance cache.

This is useful for "pre-processing" or configuration-like properties that
should be memoized at the class level (because they are expensive to compute
and identical for all instances) but still callable on instances when needed.
"""

from typing import override

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


class pre_parent_cached_property(mixed_parent_cached_property):
    """Mixed parent-cached descriptor that skips instance caching.

    Overrides ``__set__`` so that when ``node`` is an instance, the value
    is returned immediately without being written to the cache dict.  When
    ``node`` is a class, the normal parent-aware caching logic is used.
    """

    @override
    def __set__(self, node: object, value: object) -> object:
        self.get_node(node)
        if not Is.Class(node):
            # Instance access: return the fresh value, do not memoize.
            return value
        return super().__set__(node, value)


class pre_cached_property(mixed_cached_property):
    """Mixed cached descriptor that skips instance caching.

    Same semantics as ``pre_parent_cached_property`` but caches directly on
    the accessed class rather than the owning parent class.
    """

    @override
    def __set__(self, node: object, value: object) -> object:
        self.get_node(node)
        if not Is.Class(node):
            return value
        return super().__set__(node, value)

    @class_cached_property
    def here(cls) -> type[pre_parent_cached_property]:
        return pre_parent_cached_property
