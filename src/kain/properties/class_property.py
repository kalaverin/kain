"""Class-level and mixed-level property descriptors.

The descriptors here extend :class:`BaseProperty` so that the wrapped function
receives the *class* itself (for ``class_property``) or either the instance or
the class (for ``mixed_property``) as its first positional argument.

Both descriptors transparently support async functions: if ``self.function``
is a coroutine function, the result is wrapped with
:func:`asyncio.ensure_future` so the caller receives an awaitable.
"""

from __future__ import annotations

from asyncio import ensure_future
from collections.abc import Awaitable, Callable
from functools import cached_property
from inspect import iscoroutinefunction
from typing import Any, Generic, TypeVar, overload, override

from kain.internals import Is, Who, get_owner
from kain.properties.primitives import (
    AttributeException,
    BaseProperty,
    ContextFaultError,
)

__all__ = (
    "class_property",
    "mixed_property",
)

T = TypeVar("T")
R = TypeVar("R")


class class_property(BaseProperty, Generic[T, R]):
    """Descriptor that calls ``function(klass)``.

    When the attribute is accessed as ``MyClass.attr`` *or*
    ``MyClass().attr``, the underlying function is invoked with ``MyClass``
    (the class object) as the first argument.  This is analogous to a
    ``classmethod``, but for properties.

    The descriptor also resolves the *owning* class via
    :func:`kain.internals.get_owner`.  If the property is inherited, the
    ``node`` passed to ``get_owner`` ensures the MRO lookup starts from the
    concrete subclass, so mix-ins and subclassing behave correctly.
    """

    # ``klass`` is used by error-formatting utilities to indicate that this
    # descriptor expects a class-like ``node``.
    klass: bool | None = True

    @overload
    def __new__(
        cls,
        function: Callable[[type[T]], R],
    ) -> class_property[T, R]: ...
    @overload
    def __new__(cls, function: Callable[..., R]) -> class_property[Any, R]: ...
    def __new__(cls, function: Callable[..., R]) -> class_property[Any, R]:
        """Create a new class-property instance."""
        return object.__new__(cls)

    @cached_property
    def title(self) -> str:
        """Return a display title for this property."""
        return f"class descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        """Return a header string with owner context."""
        return self.footer(node)

    def get_node(self, node: object) -> type[T]:
        """Validate that ``node`` is a class and return its owner.

        :func:`get_owner` walks the MRO to find the class that actually
        defines this descriptor.  This matters when the property is used in
        a mixin or abstract base class and inherited by concrete subclasses.
        """
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)

        result = get_owner(node, self.name)
        return result if result is not None else node  # type: ignore[return-value]

    def call(self, node: object) -> R | Awaitable[R]:
        """Invoke ``self.function(node)``, wrapping coroutines if needed."""
        self.get_node(node)
        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                return ensure_future(value)
            return value

        except AttributeError as e:
            raise AttributeException(e) from e

    @overload
    def __get__(self, instance: None, klass: type[T] | None = ...) -> R: ...
    @overload
    def __get__(self, instance: object, klass: type[T] | None = ...) -> R: ...
    def __get__(
        self,
        instance: object | None,
        klass: type[T] | None = None,
    ) -> R:
        """Descriptor protocol hook — always delegates to ``call(klass)``."""
        return self.call(klass)  # type: ignore[return-value]


class mixed_property(BaseProperty, Generic[T, R]):
    """Descriptor that calls ``function(instance_or_klass)``.

    A *mixed* property works on both instances and classes:

    * ``obj.attr``   → ``function(obj)``
    * ``Cls.attr``   → ``function(Cls)``

    Unlike ``class_property``, the instance itself (not the class) is passed
    when accessed on an instance.  Like ``class_property``, ``get_owner`` is
    used to resolve the defining class when ``node`` is a class.
    """

    klass: bool | None = None

    @overload
    def __new__(cls, function: Callable[[T], R]) -> mixed_property[T, R]: ...
    @overload
    def __new__(cls, function: Callable[..., R]) -> mixed_property[Any, R]: ...
    def __new__(cls, function: Callable[..., R]) -> mixed_property[Any, R]:
        """Create a new mixed-property instance."""
        return object.__new__(cls)

    @cached_property
    def title(self) -> str:
        """Return a display title for this property."""
        return f"mixed descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        """Return a header string with owner context."""
        return self.footer(node, "mixed")

    def get_node(self, node: object) -> object:
        """Validate ``node`` is not ``None`` and resolve the owner if needed."""
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name) if Is.Class(node) else node

    def call(self, node: object) -> R | Awaitable[R]:
        """Invoke ``self.function(node)``, wrapping coroutines if needed."""
        self.get_node(node)
        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                return ensure_future(value)
            return value

        except AttributeError as e:
            raise AttributeException(e) from e

    @overload
    def __get__(self, instance: None, klass: type[T] | None = ...) -> R: ...
    @overload
    def __get__(self, instance: T, klass: type[T] | None = ...) -> R: ...
    def __get__(
        self,
        instance: object | None,
        klass: type[T] | None = None,
    ) -> R:
        """Descriptor protocol hook — delegates to ``call(instance or klass)``."""
        return self.call(instance or klass)  # type: ignore[return-value]
