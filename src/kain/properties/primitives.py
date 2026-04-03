"""Primitive building blocks for the ``kain.properties`` descriptor system.

This module defines the exception hierarchy, helper utilities, and the two
foundational descriptor classes (:class:`BaseProperty` and
:class:`bound_property`) that the rest of the package builds upon.

Key concepts
------------

* **``BaseProperty``** – Abstract-ish base for all descriptors.  It provides
  introspective attributes (``name``, ``title``, ``header``) and the
  ``with_parent`` classmethod that enables *parent-calling* properties.
* **``bound_property``** – The simplest concrete descriptor.  It replaces an
  instance method, stores the computed value in the instance ``__dict__``, and
  raises on class-level access or deletion.
* **``parent_call``** – A function wrapper used by ``with_parent``.  It walks
  the MRO to find the *parent* descriptor with the same name, extracts its
  wrapped function, evaluates it, and then passes that result as the *second*
  positional argument to the user-defined override.
* **``extract_wrapped``** – Central registry of "how to get the original user
  function back out of a descriptor".  Needed by ``parent_call`` so that it
  can invoke the parent implementation with the same ``(node, *args, **kw)``
  signature.
* **``cache``** – Thin convenience wrapper around :func:`functools.lru_cache`.
"""

from contextlib import suppress
from functools import cached_property, lru_cache, partial, wraps
from inspect import iscoroutine, iscoroutinefunction, isfunction, ismethod
from typing import Any, override

from kain.classes import Missing
from kain.internals import (
    Is,
    Who,
    get_attr,
)

__all__ = (
    "bound_property",
)

# Sentinel used throughout the package to mean "no value / not provided".
# It is deliberately truthy-false (``bool(Missing()) == False``) so it can
# be used in default-argument positions without colliding with ``None``.
Nothing = Missing()


class PropertyError(Exception):
    """Base exception for all property-related errors."""


class ContextFaultError(PropertyError):
    """Raised when a descriptor is accessed in an unsupported context.

    Examples: accessing an instance-only property on the class, or a
    class-only property on an instance, or with ``node is None``.
    """


class ReadOnlyError(PropertyError):
    """Raised when an attempt is made to delete a read-only descriptor."""


class AttributeException(PropertyError):  # noqa: N818
    """Wraps an ``AttributeError`` raised inside a property getter.

    The builtin ``property`` swallows the original traceback context in some
    situations; this wrapper preserves the cause while producing a clean
    message derived from the original exception text.

    .. note::
        The class name intentionally ends in ``Exception`` rather than
        ``Error`` (hence the ``N818`` noqa).
    """

    def __init__(self, origin: BaseException) -> None:
        self.exception: BaseException = origin
        super().__init__(self.message)

    @cached_property
    def message(self) -> str:
        """The last colon-separated segment of the original exception text."""
        return str(self.exception).rsplit(":", 1)[-1]


def cache(limit: Any = None):
    """Return an :func:`lru_cache` decorator (optionally pre-applied).

    This is a thin ergonomic wrapper:

    * ``@cache``         → ``lru_cache(maxsize=None)``
    * ``@cache(128)``    → ``lru_cache(maxsize=128)``
    * ``@cache(my_func)``→ ``lru_cache(maxsize=None)(my_func)``

    Parameters
    ----------
    limit:
        * If callable → apply ``lru_cache(maxsize=None)`` to it immediately.
        * If ``None`` → return ``lru_cache(maxsize=None)``.
        * If positive int/float → return ``lru_cache(maxsize=limit)``.

    Raises
    ------
    TypeError
        If ``limit`` is a ``classmethod``/``staticmethod`` (ordering mistake)
        or an invalid type.
    """
    function = partial(lru_cache, maxsize=None, typed=False)

    if isinstance(limit, classmethod | staticmethod):
        msg = f"can't wrap {Who.Is(limit)}, you must use @cache after it"
        raise TypeError(msg)

    for func in isfunction, iscoroutine, ismethod:
        if func(limit):
            return function()(limit)

    if limit is not None and (
        not isinstance(limit, float | int) or limit <= 0
    ):
        msg = f"limit must be None or positive integer, not {Who.Is(limit)}"
        raise TypeError(msg)

    return function(maxsize=limit) if limit else function()


def extract_wrapped(obj: object):
    """Extract the original user function from a descriptor object.

    This is the inverse operation of wrapping a function inside a descriptor.
    It is used by :func:`parent_call` so that it can invoke the parent
    implementation with the same signature the user originally wrote.

    Supported descriptor types (in order of checking):

    1. :class:`bound_property` → returns ``obj.__get__``
    2. Subclasses of :class:`BaseProperty` → returns ``obj.call``
    3. Built-in :class:`property` → returns ``obj.fget``
    4. :class:`functools.cached_property` → returns ``obj.func``

    Raises
    ------
    NotImplementedError
        If ``obj`` is not one of the supported descriptor types.
    """
    # when it's default instance-method replacer
    if Is.subclass(obj, bound_property):
        return obj.__get__

    # when it's full-featured (cached?) property
    if Is.subclass(obj, BaseProperty):
        return obj.call

    # when it's builtin @property
    if Is.subclass(obj, property):
        return obj.fget

    # when wrapped functions stored in .func
    if Is.subclass(obj, cached_property):
        return obj.func

    msg = (
        f"couldn't extract wrapped function from {Who.Is(obj)}: "
        f"replace it with @property, @cached_property, "
        f"@{Who.Is(bound_property)}, or other descriptor derived from "
        f"{Who.Is(BaseProperty)}"
    )
    raise NotImplementedError(msg)


def parent_call(func):
    """Wrap ``func`` so that it receives the parent descriptor's value first.

    This is the engine behind ``BaseProperty.with_parent``.  When a property
    is created ``with_parent``, the user function is wrapped with this
    decorator.  At runtime it does the following:

    1. Walk the MRO of ``node``'s class to find the *parent* descriptor with
       the same ``__name__`` as ``func``.
    2. Use :func:`extract_wrapped` to pull out the parent's implementation.
    3. Call that parent implementation, passing ``node`` plus any extra
       ``*args, **kw``.
    4. Call the user's ``func``, injecting the parent result as the *second*
       positional argument (right after ``node``).

    The ``index`` parameter to :func:`kain.internals.get_attr` is computed
    with the expression ``func.__name__ not in Is.classOf(node).__dict__``.
    Because ``bool`` is a subclass of ``int`` in Python, this yields:

    * ``False`` (i.e. ``0``) when the current class *overrides* the property —
      we want the first match in the MRO **excluding** the current class,
      which is the immediate parent implementation.
    * ``True`` (i.e. ``1``) when the current class *inherits* the property
      without overriding — we skip the first match (the inherited parent)
      and take the next one up the chain.

    Parameters
    ----------
    func: callable
        The user-defined override function.

    Returns
    -------
    callable
        A wrapper that supplies ``func(node, parent_value, *args, **kw)``.
    """
    @wraps(func)
    def parent_caller(node, *args, **kw):
        try:
            desc = get_attr(
                Is.classOf(node),
                func.__name__,
                exclude_self=True,
                index=func.__name__ not in Is.classOf(node).__dict__,
            )

            return func(
                node,
                extract_wrapped(desc)(node, *args, **kw),
                *args,
                **kw,
            )

        except RecursionError as e:
            msg = (
                f"{Who.Is(node)}.{func.__name__} call real {Who.Is(func)}, "
                f"couldn't reach parent descriptor; "
                f"maybe {Who.Is(func)} it's mixin of {Who.Is(node)}?"
            )
            raise RecursionError(msg) from e

    return parent_caller


def invoсation_context_check(func):
    """Decorator that validates the ``node`` context before calling ``func``.

    .. note::
        This helper is defined here for symmetry with ``descriptors.py`` but
        is **not used internally** by any class in the ``properties`` package.
        The cached-property hierarchy performs context checks inline inside
        ``get_node`` instead.

    The wrapper inspects ``self.klass``:

    * ``True``  → ``node`` must be a class.
    * ``False`` → ``node`` must *not* be a class.
    * ``None``  → ``node`` may be anything except ``None``.

    Raises
    ------
    ContextFaultError
        If the context does not match ``self.klass``.
    """
    @wraps(func)
    def context(self, node, *args, **kw):
        if (klass := self.klass) is not None and (
            node is None or klass != Is.Class(node)
        ):
            msg = f"{Who.Is(func)} exception, {self.header_with_context(node)}, {node=}"

            if node is None and not klass:
                msg = f"{msg}; looks like as non-instance invokation"
            raise ContextFaultError(msg)

        return func(self, node, *args, **kw)

    return context


class BaseProperty:
    """Abstract base class shared by all ``kain`` descriptors.

    ``BaseProperty`` does not implement ``__get__`` itself; subclasses decide
    whether the descriptor is instance-only, class-only, or mixed.  What this
    class *does* provide is:

    * Introspection helpers: ``name``, ``title``, ``header``
    * A printable ``__str__`` / ``__repr__``
    * The :meth:`with_parent` classmethod, which enables parent-aware
      properties via :func:`parent_call`.
    """

    @classmethod
    def with_parent(cls, function):
        """Create a descriptor instance that delegates to its parent MRO entry.

        The wrapped ``function`` will be called as
        ``function(node, parent_value, *args, **kw)`` where ``parent_value``
        is the result of evaluating the next descriptor up the inheritance
        chain with the same attribute name.
        """
        return cls(parent_call(function))

    def __init__(self, function) -> None:
        """Store the user-defined function that computes the property value."""
        self.function = function

    @cached_property
    def name(self) -> str:
        """The attribute name, taken from ``function.__name__``."""
        return self.function.__name__

    @cached_property
    def is_data(self) -> bool:
        """Return ``True`` if this descriptor defines ``__set__`` or ``__delete__``.

        Data descriptors (like ``bound_property`` and the cached family)
        take precedence over instance ``__dict__`` entries in the attribute
        lookup chain. Non-data descriptors (like ``class_property`` and
        ``mixed_property``) do not.
        """
        return hasattr(type(self), "__set__") or hasattr(
            type(self),
            "__delete__",
        )

    @cached_property
    def title(self) -> str:
        """Short human-readable label for this descriptor type.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    @cached_property
    def header(self) -> str:
        """A longer debug string combining ``title`` and ``function`` info."""
        try:
            return f"{self.title}({self.function!a})"
        except Exception:  # noqa: BLE001
            return f"{self.title}({Who.Is(self.function)})"

    def header_with_context(self, node: object) -> str:
        """Build an error message that includes the access context.

        Must be implemented by subclasses.
        """
        raise NotImplementedError

    def footer(self, node: object, mode: str = "undefined") -> str:
        """Format a debug line describing how the descriptor was invoked.

        If ``node`` is not ``None``, ``mode`` is auto-detected as either
        ``"instance"`` or ``"class"`` via :func:`kain.internals.Is.Class`.
        """
        if node is not None:
            mode = ("instance", "class")[Is.Class(node)]

        return (
            f"{self.header} called with "
            f"{mode} context ({Who.Addr(node)})"
        )

    @override
    def __str__(self) -> str:
        return f"<{self.header}>"

    @override
    def __repr__(self) -> str:
        return f"<{self.title}>"


class bound_property(BaseProperty):
    """Simple instance-bound descriptor (non-caching, read-only).

    ``bound_property`` is the moral equivalent of a write-once instance
    attribute.  When accessed on an instance, it calls ``function(instance)``,
    stores the result in the instance ``__dict__`` under the property's
    ``name``, and returns it.  Subsequent accesses bypass the descriptor
    entirely because Python finds the value in ``__dict__`` first.

    Accessing the property on the class (``instance is None``) raises
    :exc:`ContextFaultError`.  Deleting the property raises
    :exc:`ReadOnlyError`.

    Coroutine functions are rejected at decoration time; async properties
    must use ``@pin.native`` (i.e. :class:`cached_property`) instead.
    """

    def __init__(self, function) -> None:
        if iscoroutinefunction(function):
            msg = (
                f"{Who.Is(function)} is coroutine function, "
                "you must use @pin.native instead of just @pin"
            )
            raise TypeError(msg)
        super().__init__(function)

    @cached_property
    def title(self) -> str:
        return f"instance just-replace-descriptor {Who.Addr(self)}"

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node)

    def __get__(self, node: object, klass: object = Nothing) -> object:
        """Return the cached value from ``node.__dict__`` or compute it once.

        Parameters
        ----------
        node:
            The instance on which the attribute was accessed.  If ``None``,
            this means class-level access, which is forbidden.
        klass:
            The owner class (supplied automatically by the descriptor
            protocol).  Unused here except for error reporting when
            ``node is None``.
        """
        if node is None:
            raise ContextFaultError(self.header_with_context(klass))

        # If the value has already been computed and stored in the instance
        # dictionary, return it directly.  This is what makes the property
        # feel like a regular attribute after the first access.
        with suppress(KeyError):
            return node.__dict__[self.name]

        value = self.function(node)
        node.__dict__[self.name] = value
        return value

    def __delete__(self, node: object) -> None:
        """Prevent deletion of the descriptor value."""
        msg = f"{self.header_with_context(node)}: deleter called"
        raise ReadOnlyError(msg)
