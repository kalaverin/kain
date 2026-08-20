"""Class decorator that forwards method calls to a pivot attribute or object.

``proxy_to`` dynamically creates proxy methods on a class so that accessing
``instance.method()`` delegates to ``instance.pivot.method()`` (or to an
external object).  It is used by the descriptor system to wire attribute
forwarding without explicit boilerplate.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from logging import getLogger
from operator import attrgetter
from typing import Any, TypeVar, cast

from kain import _is, _who
from kain.classes import Nothing
from kain.internals import get_attr
from kain.properties.primitives import bound_property

__all__ = ("proxy_to",)
logger = getLogger(__name__)

T = TypeVar("T")


def proxy_to(  # noqa: PLR0915
    *mapping: Any,  # noqa: ANN401
    getter: Callable[[str], Callable[[Any], Any]] = attrgetter,
    default: Any = Nothing,  # noqa: ANN401
    pre: Callable[[Any], Any] | None = None,
    safe: bool = True,
) -> Callable[[type[T]], type[T]]:
    """Create a class decorator that proxies methods to a pivot.

    Args:
        *mapping: The first positional argument is the *pivot* (either a
            string attribute name or an object).  Subsequent positional
            arguments are method names to proxy.  The last positional argument
            may be ``None`` (to skip descriptor binding) or a callable that
            acts as a custom binder.
        getter: A callable that takes a method name and returns a getter
            callable ``(obj) -> value``.  Defaults to ``operator.attrgetter``.
        default: Fallback value returned when the pivot attribute is missing
            or ``None``.  Defaults to ``Nothing`` (raises ``AttributeError``).
        pre: Optional post-processor callable ``(value) -> result`` applied
            to the fetched value before returning it.
        safe: If ``True`` (default), raises ``TypeError`` when attempting to
            overwrite an existing public attribute on the class.

    Returns:
        A class decorator that mutates the target class in place.
    """
    if isinstance(mapping[-1], str):
        bind: Callable[[Callable[..., Any]], Any] | None = bound_property
    elif mapping[-1] is None:
        bind, mapping = (None, mapping[:-1])
    else:
        bind, mapping = (
            cast("Callable[[Callable[..., Any]], Any]", mapping[-1]),
            mapping[:-1],
        )

    def class_wrapper(cls: type[T]) -> type[T]:  # noqa: PLR0915
        if not _is.Class(cls):
            msg = f"{_who.Is(cls)} isn't a class"
            raise TypeError(msg)
        try:
            fields: list[str] = cls.__proxy_fields__  # type: ignore[attr-defined]  # pyrefly: ignore[missing-attribute]  # pyright: ignore[reportAttributeAccessIssue]
        except AttributeError:
            fields = []
            cls.__proxy_fields__ = fields  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]
        pivot = mapping[0]
        mapping_list = mapping[1:]
        if not mapping_list or (
            len(mapping_list) == 1 and (not isinstance(mapping_list[0], str))
        ):
            msg = f"empty mapping_list={mapping_list!r} for pivot={pivot!r}"
            raise ValueError(msg)
        for raw_method in mapping_list:
            method = cast("str", raw_method)
            if safe and (not method.startswith("_")) and get_attr(cls, method):
                msg = (
                    f"{_who.Is(cls)} already exists {method!a}: "
                    f"{get_attr(cls, method)}"
                )
                raise TypeError(msg)

            def wrapper(
                name: str,
                node: Any,  # noqa: ANN401
            ) -> Any:  # noqa: ANN401
                if not isinstance(pivot, str):
                    try:
                        return getattr(pivot, name)
                    except AttributeError as e:
                        msg = (
                            f"{_who.Is(node)}.{name} "
                            f"{_who.Name(getter)[:4]}-proxied -> "
                            f"{_who.Is(pivot)}.{name}, "
                            f"but the latter does not exist"
                        )
                        raise AttributeError(msg) from e
                try:
                    entity = getattr(node, pivot)
                except AttributeError as e:
                    msg = (
                        f"{_who.Is(node)}.{name} "
                        f"{_who.Name(getter)[:4]}-proxied -> "
                        f"{_who.Is(node)}.{pivot}.{name}, "
                        f"but {_who.Is(node)}.{pivot} does not exist"
                    )
                    raise AttributeError(msg) from e
                if entity is None:
                    msg = (
                        f"{_who.Is(node)}.{name} "
                        f"{_who.Name(getter)[:4]}-proxied -> "
                        f"{_who.Is(node)}.{pivot}.{name}, "
                        f"but current {_who.Is(node)}.{pivot} is None"
                    )
                    if default is Nothing:
                        raise AttributeError(msg)
                    msg = f"{msg}; return {_who.Is(default)}"
                    logger.warning(msg)
                    result = default
                else:
                    try:
                        result = getter(name)(entity)
                    except (AttributeError, KeyError) as e:
                        msg = (
                            f"{_who.Is(node)}.{name} "
                            f"{_who.Name(getter)[:4]}-proxied -> "
                            f"{_who.Is(node)}.{pivot}.{name}, "
                            f"but it does not exist ('{name}' not in "
                            f"{_who.Is(node)}.{pivot}): {_who.Is(entity)}"
                        )
                        if default is Nothing:
                            raise _is.classOf(e)(msg) from e
                        msg = f"{msg}; return {_who.Is(default)}"
                        logger.warning(msg)
                        result = default
                return partial(pre, result) if pre else result

            wrapper.__name__ = method
            wrapper.__qualname__ = f"{pivot}.{method}"
            if bind is None:
                node = cls.__dict__[pivot]
                try:
                    value = node.__dict__[method]
                except KeyError:
                    value = getattr(node, method)
            else:
                wrap = cast(
                    "Callable[..., Any]",
                    partial(wrapper, method),
                )
                wrap.__name__ = method  # type: ignore[attr-defined]
                wrap.__qualname__ = f"{pivot}.{method}"  # type: ignore[attr-defined]
                value = bind(wrap)
            fields.append(method)
            setattr(cls, method, value)
            fields.sort()
        return cls

    return class_wrapper
