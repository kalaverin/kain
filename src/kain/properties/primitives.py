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

Nothing = Missing()


class PropertyError(Exception): ...


class ContextFaultError(PropertyError): ...


class ReadOnlyError(PropertyError): ...


class AttributeException(PropertyError): # noqa: N818

    def __init__(self, origin: BaseException) -> None:
        self.exception: BaseException = origin
        super().__init__(self.message)

    @cached_property
    def message(self) -> str:
        return str(self.exception).rsplit(":", 1)[-1]


def cache(limit: Any = None):

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

    @classmethod
    def with_parent(cls, function):
        return cls(parent_call(function))

    def __init__(self, function) -> None:
        self.function = function

    @cached_property
    def name(self) -> str:
        return self.function.__name__

    @cached_property
    def title(self) -> str:
        raise NotImplementedError

    @cached_property
    def header(self) -> str:
        try:
            return f"{self.title}({self.function!a})"
        except Exception:  # noqa: BLE001
            return f"{self.title}({Who.Is(self.function)})"

    def header_with_context(self, node: object) -> str:
        raise NotImplementedError

    def footer(self, node: object, mode: str = "undefined") -> str:
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

    def __get__(self, node: object, klass: object) -> object:
        if node is None:
            raise ContextFaultError(self.header_with_context(klass))

        with suppress(KeyError):
            return node.__dict__[self.name]

        value = self.function(node)
        node.__dict__[self.name] = value
        return value

    def __delete__(self, node: object) -> None:
        msg = f"{self.header_with_context(node)}: deleter called"
        raise ReadOnlyError(msg)
