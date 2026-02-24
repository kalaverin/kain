"""Instance-level cached property descriptor.

This module defines :class:`cached_property` — the caching equivalent of
:class:`kain.properties.primitives.bound_property`.  It stores computed
values in a per-instance dictionary named ``__instance_memoized__`` and
supports parent-aware resolution (via inheritance from
:class:`class_parent_cached_property`).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from functools import cached_property as base_cached_property
from typing import Any, Generic, TypeVar, overload

from kain.internals import Is, Who
from kain.properties.cached.klass import (
    class_parent_cached_property,
)
from kain.properties.primitives import (
    ContextFaultError,
)

__all__ = ("cached_property",)

T = TypeVar("T")
R = TypeVar("R")


class cached_property(class_parent_cached_property[T, R], Generic[T, R]):
    """Instance-level cached descriptor.

    When accessed on an instance, the underlying function is called once,
    the result is stored in ``instance.__dict__["__instance_memoized__"]``,
    and subsequent accesses return the cached value.

    Access on the class itself (``instance is None``) raises
    :exc:`ContextFaultError`.

    Parent-aware resolution
    -----------------------
    Because this class inherits from ``class_parent_cached_property``,
    ``with_parent`` works out of the box: the parent descriptor up the MRO
    is located, evaluated, and its result is injected as the second argument
    to the overriding function.
    """

    @overload
    def __new__(
        cls,
        function: Callable[[T], R],
        **kw: object,
    ) -> cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> cached_property[Any, R]: ...
    def __new__(cls, *args: object, **kw: object) -> cached_property[Any, R]:
        """Create a new descriptor instance."""
        return object.__new__(cls)

    @base_cached_property
    def title(self) -> str:
        """Return a display title / header string for this property."""
        return f"instance data-descriptor {Who.Addr(self)}".strip()

    def get_node(self, node: object) -> T:
        """Assert ``node`` is an instance (not a class) and return it.

        Args:
            node: The object to validate as an instance.

        Returns:
            The validated instance.

        Raises:
            ContextFaultError: If ``node`` is ``None`` or a class.
        """
        if node is None or Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            if node is None and not self.klass:
                msg = f"{msg}; looks like a non-instance invocation"
            raise ContextFaultError(msg)
        return node  # type: ignore[return-value]

    def get_cache(self, node: object) -> dict[str, R | tuple[R, float]]:
        """Return (creating if necessary) the ``__instance_memoized__`` dict.

        The cache lives directly on ``node`` (the instance), so every
        instance has isolated memoization.

        Args:
            node: The instance whose cache should be returned.

        Returns:
            The ``__instance_memoized__`` dict for ``node``.
        """
        self.get_node(node)
        name = "__instance_memoized__"

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
        """Descriptor protocol hook — only instance access is allowed.

        Args:
            instance: The instance accessing the descriptor.
            klass: The owner class.

        Returns:
            The cached or computed value.

        Raises:
            ContextFaultError: If accessed on the class (``instance is None``).
        """
        if instance is None:
            raise ContextFaultError(self.header_with_context(klass))
        return self.call(instance)
