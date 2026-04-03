"""Mixed-level cached property descriptors.

A *mixed* property works on both instances and classes.  The cache is stored
in a dictionary whose name depends on the access context:

* Instance access → ``__instance_memoized__``
* Class access    → ``__class_memoized__``

Like the other cached families, there are two variants:

* ``mixed_cached_property`` - caches directly on the accessed node.
* ``mixed_parent_cached_property`` - resolves the owning class via
  :func:`kain.internals.get_owner` when accessed on a class.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from functools import cached_property
from typing import Any, Generic, TypeVar, overload, override

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

T = TypeVar("T")
R = TypeVar("R")


class mixed_parent_cached_property(class_parent_cached_property[T, R]):
    """Mixed cached descriptor with parent-aware class caching.

    When accessed on a class, the cache is stored on the *owning* class
    (found via ``get_owner``).  When accessed on an instance, the cache is
    stored on the instance itself (``__instance_memoized__``).
    """

    @cached_property
    def title(self) -> str:
        """Return a display title / header string for this property."""
        return f"mixed data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        """Return a display title / header string for this property."""
        return self.footer(node, "mixed")

    @override
    def get_node(self, node: object) -> type[T] | T:
        """Validate ``node`` is not ``None`` and resolve owner for classes.

        Args:
            node: The object to validate.

        Returns:
            The owning class if ``node`` is a class, otherwise ``node`` itself.
        """
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        result = get_owner(node, self.name)
        return result if result is not None and Is.Class(node) else node  # type: ignore[return-value]

    def get_cache(self, node: object) -> dict[str, R | tuple[R, float]]:
        """Return the cache dict whose name depends on whether ``node`` is a class.

        Args:
            node: The instance or class whose cache should be returned.

        Returns:
            The appropriate memoization dict for ``node``.
        """
        self.get_node(node)
        name = f'__{("instance", "class")[Is.Class(node)]}_memoized__'

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]  # type: ignore[no-any-return]

        cache: dict[str, R | tuple[R, float]] = {}
        setattr(node, name, cache)
        return cache

    @overload
    def __get__(self, instance: None, klass: type[T] | None = ...) -> R: ...
    @overload
    def __get__(self, instance: T, klass: type[T] | None = ...) -> R: ...
    def __get__(
        self,
        instance: object | None,
        klass: type[T] | None = None,
    ) -> R:
        """Descriptor protocol hook — ``call(instance or klass)``.

        Args:
            instance: The instance accessing the descriptor.
            klass: The owner class.

        Returns:
            The cached or computed value.
        """
        return self.call(instance or klass)  # type: ignore[return-value]


class mixed_cached_property(mixed_parent_cached_property[T, R], Generic[T, R]):
    """Mixed cached descriptor that caches directly on the accessed node.

    This is the plain variant: no ``get_owner`` lookup is performed.
    The cache lives either on the instance or on the concrete class that
    was used to access the attribute.
    """

    @overload
    def __new__(
        cls,
        function: Callable[[T], R],
        **kw: object,
    ) -> mixed_cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> mixed_cached_property[Any, R]: ...
    def __new__(
        cls,
        *args: object,
        **kw: object,
    ) -> mixed_cached_property[Any, R]:
        """Create a new descriptor instance."""
        return object.__new__(cls)

    @override
    def get_node(self, node: object) -> type[T] | T:
        """Validate ``node`` is not ``None`` and return it verbatim.

        Args:
            node: The object to validate.

        Returns:
            The validated node.
        """
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return node  # type: ignore[return-value]

    @class_cached_property
    def here(
        cls: type[mixed_parent_cached_property[Any, Any]],
    ) -> type[mixed_parent_cached_property[Any, Any]]:
        """Return the parent cached-property class this variant inherits from."""
        return mixed_parent_cached_property
