"""Class-level cached property descriptors.

This module implements the ``class_cached_property`` family.  Both variants
store their cache in a dictionary named ``__class_memoized__`` attached to
the class object (either the concrete class or the owning parent class,
depending on the variant).

In addition, :class:`CustomCallbackMixin` provides a pluggable cache
invalidation mechanism via the ``is_actual`` callback.
"""

from __future__ import annotations

from asyncio import ensure_future
from collections.abc import Awaitable, Callable
from contextlib import suppress
from functools import cached_property, partial
from inspect import iscoroutinefunction
from time import time
from typing import Any, Generic, TypeVar, overload, override

from kain.classes import Missing
from kain.internals import Is, Who, get_owner
from kain.properties.primitives import (
    AttributeException,
    BaseProperty,
    ContextFaultError,
    Nothing,
)

__all__ = (
    "class_cached_property",
    "class_parent_cached_property",
)

T = TypeVar("T")
R = TypeVar("R")


class CustomCallbackMixin:
    """Mixin that adds TTL and custom callback support to cached descriptors.

    The mixin is designed to be combined with a descriptor class (e.g.
    ``class_parent_cached_property``).  It exposes two factory classmethods:

    * :meth:`by` / :meth:`expired_by` - supply your own ``is_actual``
      predicate.
    * :meth:`ttl` - supply a numeric lifetime in seconds.

    ``is_actual`` signature
    -----------------------
    ``is_actual(self, node, stamp) -> bool | float``

    * When called **without** ``stamp`` (during ``__set__``), it should
      return the *new* stamp to store alongside the cached value.
    * When called **with** ``stamp`` (during ``call`` / cache lookup), it
      should return ``True`` if the cached value is still valid.

    If ``is_actual`` is ``Nothing`` (the default), the cache never expires.
    """

    @classmethod
    def by(
        cls,
        callback: Callable[[object, object, float], bool | float],
    ) -> Callable[..., Any]:
        """Return ``partial(cls, is_actual=callback)``.

        This lets users write custom factory decorators such as::

            my_property = class_cached_property.by(lambda self, node, stamp: ...)
        """
        return partial(cls, is_actual=callback)

    # Alias for ``by``; reads more naturally when the callback decides
    # *expiration* rather than *validity*.
    expired_by = by
    """Alias for :meth:`by`."""

    @classmethod
    def ttl(cls, expire: float) -> Callable[..., Any]:
        """Factory that returns a partial applying a time-to-live policy.

        Args:
            expire: Number of seconds the cached value remains valid.
                Must be a positive ``float`` or ``int``.

        Returns:
            A pre-configured ``partial(cls, is_actual=...)`` ready to be
            used as a decorator.

        Raises:
            TypeError: If ``expire`` is not a ``float`` or ``int``.
            ValueError: If ``expire`` is not a positive number.
        """
        if not isinstance(expire, float | int):
            msg = f"expire must be float or int, not {Who.Cast(expire)}"
            raise TypeError(msg)

        if expire <= 0:
            msg = f"expire must be a positive number, not {Who.Cast(expire)}"
            raise ValueError(msg)

        def is_actual(
            _self: object,
            _node: object,
            value: float | Missing = Nothing,
        ) -> bool | float:
            # ``value`` is the stored timestamp.  If it is a float, evaluate
            # whether it has expired.  Otherwise (shouldn't happen with
            # normal usage) return the current time so a new stamp is set.
            if isinstance(value, float):
                return value + expire > time()
            return time()

        return cls.by(is_actual)


class class_parent_cached_property(
    BaseProperty,
    CustomCallbackMixin,
    Generic[T, R],
):
    """Class-level cached descriptor that stores cache on the *owning* class.

    The *owning* class is determined by :func:`kain.internals.get_owner`,
    which walks the MRO to find the class that actually defines this
    descriptor.  This means that if ``Base`` defines the property and
    ``Child(Base)`` is accessed, the cache lives on ``Base`` — so all
    subclasses share the same cached value.

    Cache structure
    ---------------
    The cache dict is stored in the class ``__dict__`` under the key
    ``__class_memoized__``.  Each entry is keyed by the property ``name``.

    * Without ``is_actual`` → ``cache[name] = value``
    * With ``is_actual``    → ``cache[name] = (value, stamp)``
    """

    is_actual: Callable[[object, object, float], bool | float] | Missing

    @overload
    def __new__(
        cls,
        function: Callable[[type[T]], R],
        **kw: object,
    ) -> class_parent_cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> class_parent_cached_property[Any, R]: ...
    def __new__(
        cls,
        *args: object,
        **kw: object,
    ) -> class_parent_cached_property[Any, R]:
        """Create a new descriptor instance."""
        return object.__new__(cls)

    @cached_property
    def title(self) -> str:
        """Return a display title / header string for this property."""
        return f"class data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        """Return a display title / header string for this property."""
        return self.footer(node)

    def get_node(self, node: object) -> type[T]:
        """Assert ``node`` is a class and return the owning class.

        Args:
            node: The object to validate as a class.

        Returns:
            The owning class (or ``node`` itself if no owner is found).
        """
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        result = get_owner(node, self.name)
        return result if result is not None else node  # type: ignore[return-value]

    def __init__(
        self,
        function: Callable[..., R],
        is_actual: (
            Callable[[object, object, float], bool | float] | Missing
        ) = Nothing,
    ) -> None:
        """Initialize the descriptor.

        Args:
            function: The callable that computes the cached value.
            is_actual: Optional callback that validates or timestamps cache
                entries.

        Raises:
            TypeError: If a subclass defines an ``is_actual`` method and an
                ``is_actual`` keyword argument is also provided.
        """
        super().__init__(function)

        # If a subclass defines a *class-level* ``is_actual`` method, that
        # method takes precedence.  We raise if the caller also passed an
        # explicit ``is_actual`` keyword argument to avoid ambiguity.
        if method := getattr(Is.classOf(self), "is_actual", None):
            if is_actual:
                msg = (
                    f"{Who.Is(self)}.is_actual method ({Who.Cast(method)}) "
                    f"can't be overridden by the is_actual keyword argument"
                )
                raise TypeError(msg)
            is_actual = method  # type: ignore[assignment]

        self.is_actual = is_actual

    def get_cache(self, node: object) -> dict[str, R | tuple[R, float]]:
        """Return (creating if necessary) the ``__class_memoized__`` dict.

        Args:
            node: The class whose cache should be returned.

        Returns:
            The ``__class_memoized__`` dict for ``node``.
        """
        self.get_node(node)
        name = "__class_memoized__"

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]  # type: ignore[no-any-return]

        cache: dict[str, R | tuple[R, float]] = {}
        setattr(node, name, cache)
        return cache

    def call(self, node: object) -> R | Awaitable[R]:
        """Look up the cached value or compute, cache, and return it.

        When ``is_actual`` is configured, the cache entry is a 2-tuple
        ``(value, stamp)``.  The callback is invoked as
        ``self.is_actual(self, pivot, stamp)`` and the value is reused
        only when the callback returns *exactly* ``True``.

        Args:
            node: The class on which to look up or compute the value.

        Returns:
            The cached or newly computed value.

        Raises:
            AttributeException: If computing the value raises
                :exc:`AttributeError`.
        """
        pivot = self.get_node(node)
        with suppress(KeyError):

            stored = self.get_cache(pivot)[self.name]
            if not self.is_actual:
                return stored  # type: ignore[return-value]

            value, stamp = stored  # type: ignore[misc]
            if self.is_actual(self, pivot, stamp) is True:
                return value

        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                value = ensure_future(value)

        except AttributeError as e:
            raise AttributeException(e) from e

        return self.__set__(pivot, value)

    @overload
    def __get__(self, instance: None, klass: type[T] | None = ...) -> R: ...
    @overload
    def __get__(self, instance: object, klass: type[T] | None = ...) -> R: ...
    def __get__(
        self,
        instance: object | None,
        klass: type[T] | None = None,
    ) -> R:
        """Always invoke ``call(klass)`` — class-level access only.

        Args:
            instance: The instance accessing the descriptor (always ``None``).
            klass: The owner class.

        Returns:
            The cached or computed value.
        """
        return self.call(klass)  # type: ignore[return-value]

    def __set__(self, node: object, value: R) -> R:
        """Store ``value`` in the class cache, optionally with a timestamp.

        Args:
            node: The class on which to store the value.
            value: The value to cache.

        Returns:
            The cached value.
        """
        cache = self.get_cache(node)

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value

    def __delete__(self, node: object) -> None:
        """Remove the cached entry for this property name.

        Args:
            node: The class from which to remove the cached entry.
        """
        cache = self.get_cache(node)
        with suppress(KeyError):
            del cache[self.name]


class class_cached_property(class_parent_cached_property[T, R]):
    """Class-level cached descriptor that stores cache on the accessed class.

    This is the "plain" variant of ``class_parent_cached_property``.  Rather
    than resolving the *owning* parent via ``get_owner``, it caches directly
    on ``node`` (the class that was passed to ``__get__``).  Consequently,
    each subclass maintains its own independent cache.
    """

    @overload
    def __new__(
        cls,
        function: Callable[[type[T]], R],
        **kw: object,
    ) -> class_cached_property[T, R]: ...
    @overload
    def __new__(
        cls,
        function: Callable[..., R],
        **kw: object,
    ) -> class_cached_property[Any, R]: ...
    def __new__(
        cls,
        *args: object,
        **kw: object,
    ) -> class_cached_property[Any, R]:
        """Create a new descriptor instance."""
        return object.__new__(cls)

    @override
    def get_node(self, node: object) -> type[T]:
        """Assert ``node`` is a class and return it verbatim (no owner lookup).

        Args:
            node: The object to validate as a class.

        Returns:
            The validated class.
        """
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return node  # type: ignore[return-value]

    # Self-referential class property that exposes the parent class.
    # Useful for introspection and for symmetry with the other cached
    # descriptor families (mixed, pre, post).
    @class_parent_cached_property
    def here(
        cls: type[class_parent_cached_property[Any, Any]],
    ) -> type[class_parent_cached_property[Any, Any]]:
        """Return the parent cached-property class this variant inherits from."""
        return class_parent_cached_property
