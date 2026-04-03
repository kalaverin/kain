"""Class-level cached property descriptors.

This module implements the ``class_cached_property`` family.  Both variants
store their cache in a dictionary named ``__class_memoized__`` attached to
the class object (either the concrete class or the owning parent class,
depending on the variant).

In addition, :class:`CustomCallbackMixin` provides a pluggable cache
invalidation mechanism via the ``is_actual`` callback.
"""

from asyncio import ensure_future
from collections.abc import Callable
from contextlib import suppress
from functools import cached_property, partial
from inspect import iscoroutinefunction
from time import time
from typing import override

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


class CustomCallbackMixin:
    """Mixin that adds TTL and custom callback support to cached descriptors.

    The mixin is designed to be combined with a descriptor class (e.g.
    ``class_parent_cached_property``).  It exposes two factory classmethods:

    * :meth:`by` / :meth:`expired_by` – supply your own ``is_actual``
      predicate.
    * :meth:`ttl` – supply a numeric lifetime in seconds.

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
    def by(cls, callback):
        """Return ``partial(cls, is_actual=callback)``.

        This lets users write custom factory decorators such as::

            my_property = class_cached_property.by(lambda self, node, stamp: ...)
        """
        return partial(cls, is_actual=callback)

    # Alias for ``by``; reads more naturally when the callback decides
    # *expiration* rather than *validity*.
    expired_by = by

    @classmethod
    def ttl(cls, expire: float):
        """Factory that returns a partial applying a time-to-live policy.

        Parameters
        ----------
        expire:
            Number of seconds the cached value remains valid.  Must be
            a positive ``float`` or ``int``.

        Returns
        -------
        partial
            A pre-configured ``partial(cls, is_actual=...)`` ready to be
            used as a decorator.
        """
        if not isinstance(expire, float | int):
            msg = f"expire must be float or int, not {Who.Cast(expire)}"
            raise TypeError(msg)

        if expire <= 0:
            msg = f"expire must be positive number, not {Who.Cast(expire)}"
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


class class_parent_cached_property(BaseProperty, CustomCallbackMixin):
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

    @cached_property
    def title(self) -> str:
        return f"class data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node)

    def get_node(self, node: object) -> object:
        """Assert ``node`` is a class and return the owning class."""
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name)

    def __init__(self, function: Callable, is_actual=Nothing):
        super().__init__(function)

        # If a subclass defines a *class-level* ``is_actual`` method, that
        # method takes precedence.  We raise if the caller also passed an
        # explicit ``is_actual`` keyword argument to avoid ambiguity.
        if method := getattr(Is.classOf(self), "is_actual", None):
            if is_actual:
                msg = (
                    f"{Who.Is(self)}.is_actual method ({Who.Cast(method)}) "
                    f"can't override by is_actual kw: {Who.Cast(is_actual)}"
                )
                raise TypeError(msg)
            is_actual = method

        self.is_actual = is_actual

    def get_cache(self, node: object) -> dict[str, object]:
        """Return (creating if necessary) the ``__class_memoized__`` dict."""
        self.get_node(node)
        name = "__class_memoized__"

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def call(self, node: object) -> object:
        """Lookup the cached value or compute, cache, and return it.

        When ``is_actual`` is configured, the cache entry is a 2-tuple
        ``(value, stamp)``.  The callback is invoked as
        ``self.is_actual(self, pivot, stamp)`` and the value is reused
        only when the callback returns *exactly* ``True``.
        """
        pivot = self.get_node(node)
        with suppress(KeyError):

            stored = self.get_cache(pivot)[self.name]
            if not self.is_actual:
                return stored

            value, stamp = stored
            if self.is_actual(self, pivot, stamp) is True:
                return value

        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                value = ensure_future(value)

        except AttributeError as e:
            raise AttributeException(e) from e

        return self.__set__(pivot, value)

    def __get__(self, instance: object, klass: object) -> object:
        """Always invoke ``call(klass)`` — class-level access only."""
        return self.call(klass)

    def __set__(self, node: object, value: object) -> object:
        """Store ``value`` in the class cache, optionally with a timestamp."""
        cache = self.get_cache(node)

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value

    def __delete__(self, node: object) -> None:
        """Remove the cached entry for this property name."""
        cache = self.get_cache(node)
        with suppress(KeyError):
            del cache[self.name]


class class_cached_property(class_parent_cached_property):
    """Class-level cached descriptor that stores cache on the accessed class.

    This is the "plain" variant of ``class_parent_cached_property``.  Rather
    than resolving the *owning* parent via ``get_owner``, it caches directly
    on ``node`` (the class that was passed to ``__get__``).  Consequently,
    each subclass maintains its own independent cache.
    """

    @override
    def get_node(self, node: object) -> object:
        """Assert ``node`` is a class and return it verbatim (no owner lookup)."""
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return node

    # Self-referential class property that exposes the parent class.
    # Useful for introspection and for symmetry with the other cached
    # descriptor families (mixed, pre, post).
    @class_parent_cached_property
    def here(cls) -> type[class_parent_cached_property]:
        return class_parent_cached_property
