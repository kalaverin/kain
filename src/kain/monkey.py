"""Monkey-patching utilities for modules, classes, and functions.

This module provides a small toolkit for runtime modification of
existing objects:

    - ``Monkey.expect`` – decorator that silences specified exceptions.
    - ``Monkey.patch`` – replace an attribute on a module or object.
    - ``Monkey.bind`` – attach a new function to a target object.
    - ``Monkey.wrap`` – wrap an existing callable with custom logic.

Example:
    >>> from kain.monkey import Monkey
    >>> import os
    >>> original = os.path.join
    >>> def my_join(*args):
    ...     return "joined:" + original(*args)
    ...
    >>> Monkey.patch((os.path, "join"), my_join)
    <function my_join at ...>
    >>> Monkey.mapping[my_join] is original
    True
"""

from collections.abc import Callable
from contextlib import suppress
from functools import wraps
from logging import getLogger
from types import ModuleType
from typing import Any, ClassVar, cast

from kain.importer import required
from kain.internals import Is, Who

logger = getLogger(__name__)


class Monkey:
    """Namespace for monkey-patching helpers."""

    mapping: ClassVar[dict[object, object]] = {}
    """Maps patched objects back to their original values."""

    @classmethod
    def expect(
        cls,
        *exceptions: type[BaseException],
    ) -> Callable[[Callable[..., object]], Any]:
        """Return a decorator that suppresses the given exceptions.

        The decorator is intended for classmethods: it turns the wrapped
        function into a classmethod and silently ignores any of the
        specified exceptions raised inside it.

        Args:
            *exceptions: Exception types to catch and suppress.

        Returns:
            A decorator that accepts a function and returns a
            classmethod wrapper.

        Example:
            >>> class Kls:
            ...     @Monkey.expect(ValueError)
            ...     def parse(cls, data: str) -> int:
            ...         return int(data)
            ...
            >>> Kls.parse("not-a-number")  # no exception raised
        """

        def make_wrapper(func: Callable[..., object]) -> Any:
            @wraps(func)
            def wrapper(
                klass: type[object],
                *args: object,
                **kw: object,
            ) -> object:
                with suppress(*exceptions):
                    return func(klass, *args, **kw)
                return None

            return classmethod(wrapper)  # type: ignore[arg-type]

        return make_wrapper

    @classmethod
    def patch(
        cls,
        module: str | ModuleType | tuple[object, str],
        new: object,
    ) -> object:
        """Replace an attribute on *module* with *new*.

        The original value is stored in :attr:`Monkey.mapping` so it can
        be restored later if needed.

        Args:
            module: One of:
                - A dotted import path (e.g. ``"os.path.join"``)
                - A module object (``name`` is taken from ``new.__name__``)
                - A ``(node, name)`` tuple pointing to the attribute
            new: The replacement object.

        Returns:
            The object that was actually set (usually *new*).

        Raises:
            ImportError: If the dotted path cannot be resolved.
            RuntimeError: If the old and new values are identical after
                assignment.
        """
        node: object
        name: str

        if isinstance(module, tuple):
            node, name = module

        elif Is.module(module):
            node, name = module, cast("str", getattr(new, "__name__", ""))

        else:
            path, name = module.rsplit(".", 1)
            try:
                node = required(path)
            except ImportError:
                logger.error(f"{module=} import error")  # noqa: TRY400
                raise

        if getattr(node, name, None) is new:
            return new

        old = required(cast("str", node), name) if Who.Is(node) != name else node

        setattr(node, name, new)
        new = getattr(node, name)
        if old is new:
            raise RuntimeError

        cls.mapping[new] = old
        logger.debug(f"{Who.Addr(old)} -> {Who.Addr(new)}")
        return new

    @classmethod
    def bind(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Bind *func* as an attribute of *node*.

        Args:
            node: Target object or a dotted import path resolving to it.
            name: Attribute name to use. Defaults to ``func.__name__``.
            decorator: Optional decorator to apply. When this is exactly
                :class:`classmethod`, *node* is injected as the first
                positional argument.

        Returns:
            A decorator that binds the wrapped function to *node*.
        """
        node = required(node) if isinstance(node, str) else node

        def bind(func: Callable[..., object]) -> Callable[..., object]:
            @wraps(func)
            def wrapper(*args: object, **kw: object) -> object:
                if decorator is classmethod:
                    return func(node, *args, **kw)
                return func(*args, **kw)

            local = name or cast("str", getattr(func, "__name__", ""))
            setattr(node, local, wrapper)
            logger.info(f"{Who.Is(node)}.{local} <- {Who.Addr(func)}")
            return wrapper

        return bind

    @classmethod
    def wrap(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        """Wrap an existing callable on *node*.

        The wrapper receives the original callable as its first argument,
        followed by the normal positional and keyword arguments.

        Args:
            node: Target object or a dotted import path resolving to it.
            name: Name of the attribute to wrap. Defaults to
                ``func.__name__``.
            decorator: Optional decorator to apply to the wrapper before
                patching.

        Returns:
            A decorator that returns the wrapper function.
        """
        node = required(node) if isinstance(node, str) else node

        def wrap(func: Callable[..., object]) -> Callable[..., object]:
            wrapped_name = name or cast("str", getattr(func, "__name__", ""))
            if Who.Name(node) != wrapped_name:
                wrapped_func = required(cast("str", node), wrapped_name)
            else:
                wrapped_func = node

            @wraps(func)
            def wrapper(*args: object, **kw: object) -> object:
                return func(wrapped_func, *args, **kw)

            logger.info(f"{Who.Is(node)}.{wrapped_name} <- {Who.Addr(func)}")

            wrapped = decorator(wrapper) if decorator else wrapper
            _ = cls.patch((node, wrapped_name), wrapped)
            return wrapper

        return wrap
