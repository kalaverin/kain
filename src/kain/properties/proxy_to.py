"""Class decorator for proxying attributes from a nested object or descriptor.

``proxy_to`` lets a class forward selected attributes to an inner object
(accessed via an instance attribute or a class-level descriptor) without
writing boilerplate properties or methods.

Typical usage::

    @proxy_to("engine", "start", "stop")
    class Car:
        def __init__(self):
            self.engine = Engine()

    car = Car()
    car.start()   # calls car.engine.start()
    car.stop()    # calls car.engine.stop()

The decorator supports three binding strategies:

1. **Descriptor binding** (default) — wraps each proxy in ``bound_property``,
   caching the result in the instance ``__dict__``.
2. **Direct copy** — pass ``None`` as the last positional argument to copy
   the attribute reference directly from a class-level pivot descriptor.
3. **Custom binding** — pass a callable (e.g. a descriptor class) as the last
   positional argument to wrap each proxy with custom logic.
"""

# ruff: noqa: ANN401

from __future__ import annotations

from functools import partial
from logging import getLogger
from operator import attrgetter
from typing import TYPE_CHECKING

from kain.classes import Nothing
from kain.internals import (
    Is,
    Who,
    get_attr,
)
from kain.properties.primitives import bound_property

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

__all__ = ("proxy_to",)

logger = getLogger(__name__)


def proxy_to(  # noqa: PLR0915
    *mapping: object,
    getter: Callable[[str], Callable[[Any], Any]] = attrgetter,
    default: Any = Nothing,
    pre: Callable[[Any], Any] | None = None,
    safe: bool = True,
) -> Callable[[type[Any]], type[Any]]:
    """Forward attributes from a *pivot* object onto the decorated class.

    Args:
        *mapping: Variable-length positional arguments.
            The first element is the **pivot** — either a ``str`` naming an
            attribute on the instance (e.g. ``"inner"``), or an arbitrary
            object used directly as the source. All remaining elements
            (except an optional trailing bind specifier) are the names of
            attributes to proxy.
            The last element may optionally control binding: if it is a
            ``str``, it is treated as an attribute name and binding defaults
            to :class:`bound_property`; ``None`` means direct copy mode;
            any other callable is treated as a custom descriptor factory.
        getter: Callable that receives an attribute name and returns a
            callable able to extract that attribute from the pivot entity.
            Defaults to :func:`operator.attrgetter`.
        default: Fallback value returned when the pivot is ``None`` or the
            proxied attribute is missing. Defaults to the sentinel
            ``Nothing``; in that case an appropriate exception is raised
            instead of returning a fallback.
        pre: Optional callable applied to the result (despite the name
            ``pre``, it acts as post-processing). When provided, the proxy
            result is wrapped with :func:`functools.partial(pre, result)`,
            so the attribute access returns a callable that must be invoked
            to obtain the final value.
        safe: If ``True`` (default), raises ``TypeError`` when attempting
            to proxy a public attribute (name not starting with ``_``) that
            already exists on the class. Set to ``False`` to allow
            overwriting existing attributes.

    Returns:
        A class decorator that installs the proxies and returns the class.

    Raises:
        TypeError: If the decorated object is not a class, or if
            ``safe=True`` and a public attribute already exists on the class.
        ValueError: If no attributes are requested for proxying.
        AttributeError: If the pivot is missing, the pivot is ``None`` and
            no ``default`` is provided, or the proxied attribute does not
            exist on the pivot entity and no ``default`` is provided.

    Example:
        Proxy instance attributes with default descriptor binding::

            @proxy_to("inner", "foo", "bar")
            class Wrapper:
                def __init__(self):
                    self.inner = Inner()

        Direct copy from a class-level descriptor::

            @proxy_to("desc", "method", None)
            class Wrapper:
                desc = SomeDescriptor()

        Custom getter for dict-like pivots::

            @proxy_to("data", "name", getter=itemgetter)
            class Wrapper:
                def __init__(self):
                    self.data = {"name": "Alice"}
    """
    # Determine the binding strategy from the last positional argument.
    bind: Callable[[Callable[[Any], Any]], Any] | None
    if isinstance(mapping[-1], str):
        # Last arg is a method name → default to bound_property binding.
        bind = bound_property

    elif mapping[-1] is None:
        # Explicit ``None`` requests direct copy mode.
        bind, mapping = None, mapping[:-1]

    else:
        # Anything else is treated as a custom descriptor factory.
        bind, mapping = mapping[-1], mapping[:-1]

    def class_wrapper(cls: type[Any]) -> type[Any]:  # noqa: PLR0915
        """Build a class-level forwarding wrapper."""
        if not Is.Class(cls):
            msg = f"{Who.Is(cls)} isn't a class"
            raise TypeError(msg)

        # Maintain a sorted list of proxied fields on the class.
        try:
            fields: list[str] = cls.__proxy_fields__
        except AttributeError:
            fields = []
            cls.__proxy_fields__ = fields

        pivot: str | Any = mapping[0]
        mapping_list: tuple[Any, ...] = mapping[1:]

        # Validate that at least one attribute name was given.
        if not mapping_list or (
            len(mapping_list) == 1 and not isinstance(mapping_list[0], str)
        ):
            raise ValueError(f"empty {mapping_list=} for {pivot=}")

        for method in mapping_list:
            # Collision guard: refuse to overwrite existing public attributes
            # unless ``safe=False`` was requested.
            if safe and not method.startswith("_") and get_attr(cls, method):
                msg = (
                    f"{Who.Is(cls)} already exists {method!a}: "
                    f"{get_attr(cls, method)}"
                )
                raise TypeError(msg)

            def wrapper(name: str, node: Any) -> Any:
                """Build an instance-level forwarding wrapper."""
                # When ``pivot`` is not a string, it is an object used
                # directly as the source, ignoring ``node`` entirely.
                if not isinstance(pivot, str):
                    try:
                        return getattr(pivot, name)
                    except AttributeError as e:
                        msg = (
                            f"{Who.Is(node)}.{name} "
                            f"{Who.Name(getter)[:4]}-proxied -> "
                            f"{Who.Is(pivot)}.{name}, but the latter does not exist"
                        )
                        raise AttributeError(msg) from e

                # Resolve the pivot attribute on the accessing instance.
                try:
                    entity = getattr(node, pivot)
                except AttributeError as e:
                    msg = (
                        f"{Who.Is(node)}.{name} "
                        f"{Who.Name(getter)[:4]}-proxied -> "
                        f"{Who.Is(node)}.{pivot}.{name}, but "
                        f"{Who.Is(node)}.{pivot} does not exist"
                    )
                    raise AttributeError(msg) from e

                if entity is None:
                    msg = (
                        f"{Who.Is(node)}.{name} "
                        f"{Who.Name(getter)[:4]}-proxied -> "
                        f"{Who.Is(node)}.{pivot}.{name}, but current "
                        f"{Who.Is(node)}.{pivot} is None"
                    )

                    if default is Nothing:
                        raise AttributeError(msg)

                    msg = f"{msg}; return {Who.Is(default)}"
                    logger.warning(msg)
                    result = default

                else:
                    # Extract the requested attribute from the pivot entity.
                    try:
                        result = getter(name)(entity)

                    except (AttributeError, KeyError) as e:
                        msg = (
                            f"{Who.Is(node)}.{name} "
                            f"{Who.Name(getter)[:4]}-proxied -> "
                            f"{Who.Is(node)}.{pivot}.{name}, but it does not exist "
                            f"('{name}' not in {Who.Is(node)}.{pivot}): "
                            f"{Who.Is(entity)}"
                        )

                        if default is Nothing:
                            raise Is.classOf(e)(msg) from e

                        msg = f"{msg}; return {Who.Is(default)}"
                        logger.warning(msg)
                        result = default

                # Optional post-processing via partial application.
                return partial(pre, result) if pre else result

            wrapper.__name__ = method
            wrapper.__qualname__ = f"{pivot}.{method}"

            value: Any
            if bind is None:
                # Direct copy mode: look up the attribute on the class-level
                # pivot descriptor and copy the reference verbatim.
                node = cls.__dict__[pivot]
                try:
                    value = node.__dict__[method]
                except KeyError:
                    value = getattr(node, method)
            else:
                # Bind the wrapper with the chosen descriptor factory.
                wrap: Callable[[Any], Any] = partial(wrapper, method)
                wrap.__name__ = method
                wrap.__qualname__ = f"{pivot}.{method}"
                value = bind(wrap)

            fields.append(method)
            setattr(cls, method, value)
            cls.__proxy_fields__.sort()

        return cls

    return class_wrapper
