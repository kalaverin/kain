from __future__ import annotations

from functools import wraps
from logging import getLogger
from typing import (
    TYPE_CHECKING,
    ClassVar,
    Concatenate,
    ParamSpec,
    TypeVar,
)

from kain import _is, _who
from kain.classes import Missing
from kain.importer import required

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

_P = ParamSpec("_P")
_T = TypeVar("_T")

Nothing = Missing()

logger = getLogger(__name__)


def get_name(node: object) -> str:
    if (
        _is.Class(node)
        or _is.builtin(node)
        or _is.function(node)
        or _is.method(node)
        or _is.module(node)
    ):
        return _who.Name(node)

    name = _who.Cast(node)
    msg = "cannot derive attribute name"
    logger.error(
        msg,
        extra={
            "object": name,
        },
    )
    raise AttributeError(msg)


def get_attribute(
    node: object,
    name: str,
    *,
    by_short: bool = False,
) -> object:
    if (_who.Name(node) if by_short else _who.Is(node)) == name:
        return node
    return getattr(node, name)


def import_from_string(node: str | object) -> object:
    if isinstance(node, str):
        return required(node)
    return node


def parse_target(
    target: str | ModuleType | tuple[object, str] | object,
    replacement: object | None = None,
) -> tuple[object, str]:
    """Normalize a patch/wrap target into ``(node, name)``.

    Args:
        target: One of:
            - A dotted import path (e.g. ``"os.path.join"``)
            - A module object (``name`` comes from *replacement*'s
              short name)
            - A ``(node, name)`` tuple
            - An arbitrary object (``name`` comes from *replacement*'s
              short name)
        replacement: The replacement object. Its short name is used as a
            fallback when *target* does not explicitly specify one.

    Returns:
        A ``(node, name)`` tuple.

    Raises:
        ImportError: If a dotted path cannot be resolved.
        AttributeError: If *replacement* has no usable short name and
            the target requires one.
    """
    if isinstance(target, tuple):
        return target

    if _is.module(target):
        return target, get_name(replacement)

    if isinstance(target, str):
        try:
            if "." in target:
                parent_path, name = target.rsplit(".", 1)
                return required(parent_path), name

            return required(target), get_name(replacement)

        except ImportError:
            logger.error("target=%r import error", target)  # noqa: TRY400
            raise

    return target, get_name(replacement)


class Monkey:
    """Namespace for reversible runtime attribute patching utilities."""

    mapping: ClassVar[dict[object, object]] = {}

    @classmethod
    def replace(
        cls,
        target: str | ModuleType | tuple[object, str],
        replacement: _T,
    ) -> _T:
        """Replace an attribute on *target* with *replacement*.

        The original value is stored in :attr:`Monkey.mapping` so it can
        be restored later if needed.

        Args:
            target: One of:
                - A dotted import path (e.g. ``"os.path.join"``)
                - A module object (``name`` is taken from *replacement*'s
                  short name)
                - A ``(node, name)`` tuple pointing to the attribute
            replacement: The replacement object.

        Returns:
            The object that was actually set.

        Raises:
            ImportError: If the dotted path cannot be resolved.
            RuntimeError: If the old and new values are identical after
                assignment.
        """
        node, name = parse_target(target, replacement=replacement)
        if getattr(node, name, Nothing) is replacement:
            return replacement

        old = get_attribute(node, name)

        setattr(node, name, replacement)
        set_new = getattr(node, name)

        if old is set_new:
            msg = "patch failed: old and new are identical after assignment"
            logger.error(
                msg,
                extra={
                    "target": _who.Cast(target),
                    "replacement": _who.Cast(replacement),
                },
            )
            raise RuntimeError(msg)

        cls.mapping[set_new] = old
        logger.debug(
            "attribute replaced",
            extra={
                "before": _who.Addr(old),
                "after": _who.Addr(set_new),
            },
        )
        return set_new

    @classmethod
    def bind(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., _T]], Callable[_P, _T]]:
        """Bind *func* as an attribute of *node*.

        Args:
            node: Target object or a dotted import path resolving to it.
            name: Attribute name to use. Defaults to the short name of
                *func*.
            decorator: Optional decorator to apply. When this is exactly
                :class:`classmethod`, *node* is injected as the first
                positional argument.

        Returns:
            A decorator that binds the wrapped function to *node*.

        Raises:
            ImportError: If *node* is a string and ``required`` fails to
                resolve it.
            AttributeError: If ``setattr`` on *node* fails.
        """
        target = import_from_string(node)

        def bind(func: Callable[..., _T]) -> Callable[_P, _T]:
            @wraps(func)
            def wrapper(*args: _P.args, **kw: _P.kwargs) -> _T:
                if decorator is classmethod:
                    return func(target, *args, **kw)
                return func(*args, **kw)

            local_name = name or get_name(func)
            setattr(target, local_name, wrapper)
            logger.debug(
                "attribute bounded with",
                extra={
                    "before": f"{_who.Is(target)}.{local_name}",
                    "after": _who.Addr(func),
                },
            )
            return wrapper

        return bind

    @classmethod
    def wrap(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[
        [Callable[Concatenate[object, _P], _T]],
        Callable[_P, _T],
    ]:
        """Wrap an existing callable on *node*.

        The wrapper receives the original callable as its first argument,
        followed by the normal positional and keyword arguments.

        Args:
            node: Target object or a dotted import path resolving to it.
            name: Name of the attribute to wrap. Defaults to the short
                name of *func*.
            decorator: Optional decorator to apply to the wrapper before
                patching.

        Returns:
            A decorator that returns the wrapper function.

        Raises:
            ImportError: If *node* is a string and ``required`` fails to
                resolve it.
        """

        def wrap(
            func: Callable[Concatenate[object, _P], _T],
        ) -> Callable[_P, _T]:
            wrapped_name = name or get_name(func)
            target = import_from_string(node)
            original = get_attribute(target, wrapped_name, by_short=True)

            @wraps(func)
            def wrapper(*args: _P.args, **kw: _P.kwargs) -> _T:
                return func(original, *args, **kw)

            logger.debug(
                "attribute wrapped with",
                extra={
                    "before": f"{_who.Is(target)}.{wrapped_name}",
                    "after": _who.Addr(func),
                },
            )
            to_patch = decorator(wrapper) if decorator else wrapper
            cls.replace((target, wrapped_name), to_patch)
            return wrapper

        return wrap
