"""Mixed-level cached property descriptors.

A *mixed* property works on both instances and classes.  The cache is stored
in a dictionary whose name depends on the access context:

* Instance access → ``__instance_memoized__``
* Class access    → ``__class_memoized__``

Like the other cached families, there are two variants:

* ``mixed_cached_property`` – caches directly on the accessed node.
* ``mixed_parent_cached_property`` – resolves the owning class via
  :func:`kain.internals.get_owner` when accessed on a class.
"""

from contextlib import suppress
from functools import cached_property
from typing import override

from kain.internals import Is, Who, get_owner
from kain.properties.cached.klass import (
    class_cached_property,
    class_parent_cached_property,
)
from kain.properties.primitives import ContextFaultError

__all__ = (
    "mixed_cached_property",
    "mixed_parent_cached_property",
)


class mixed_parent_cached_property(class_parent_cached_property):
    """Mixed cached descriptor with parent-aware class caching.

    When accessed on a class, the cache is stored on the *owning* class
    (found via ``get_owner``).  When accessed on an instance, the cache is
    stored on the instance itself (``__instance_memoized__``).
    """

    @cached_property
    def title(self) -> str:
        return f"mixed data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node, "mixed")

    @override
    def get_node(self, node: object) -> object:
        """Validate ``node`` is not ``None`` and resolve owner for classes."""
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name) if Is.Class(node) else node

    def get_cache(self, node: object) -> dict[str, object]:
        """Return the cache dict whose name depends on ``node`` being a class."""
        self.get_node(node)
        name = f'__{("instance", "class")[Is.Class(node)]}_memoized__'

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def __get__(self, instance: object, klass: object) -> object:
        """Descriptor protocol hook — ``call(instance or klass)``."""
        return self.call(instance or klass)


class mixed_cached_property(mixed_parent_cached_property):
    """Mixed cached descriptor that caches directly on the accessed node.

    This is the plain variant: no ``get_owner`` lookup is performed.
    The cache lives either on the instance or on the concrete class that
    was used to access the attribute.
    """

    @override
    def get_node(self, node: object) -> object:
        """Validate ``node`` is not ``None`` and return it verbatim."""
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return node

    @class_cached_property
    def here(cls) -> type[mixed_parent_cached_property]:
        return mixed_parent_cached_property
