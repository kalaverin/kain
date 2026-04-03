from contextlib import suppress
from functools import cached_property, partial
from time import time
from typing import ClassVar, override

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
    "mixed_cached_property",
)


class CustomCallbackMixin:

    @classmethod
    def expired_by(cls, callback):
        return partial(cls, is_actual=callback)

    @classmethod
    def ttl(cls, expire: float):

        if expire <= 0:
            msg = f"expire must be positive number, not {Who.Cast(expire)}"
            raise ValueError(msg)

        def is_actual(
            *_: object,
            value: float | Missing = Nothing,
        ) -> bool | float:
            if isinstance(value, float):
                return value + expire > time()
            return time()

        return cls.expired_by(is_actual)


class class_cached_property(BaseProperty):

    klass: ClassVar[bool | None] = True

    @cached_property
    def title(self) -> str:
        return f"class data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node)

    def get_node(self, node: object) -> object:
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name)

    def __init__(self, function: Callable, is_actual=Nothing) -> None:
        super().__init__(function)

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
        self.get_node(node)
        name = "__class_memoized__"

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def call(self, node: object) -> object:
        self.get_node(node)

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

        except AttributeError as e:
            raise AttributeException(e) from e

        return self.__set__(pivot, value)

    def __get__(self, instance: object, klass: object) -> object:
        return self.call(klass)

    def __set__(self, node: object, value: object) -> object:
        cache = self.get_cache(node)

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value

    def __delete__(self, node: object) -> None:
        cache = self.get_cache(node)
        with suppress(KeyError):
            del cache[self.name]


class mixed_cached_property(BaseProperty):

    klass: ClassVar[bool | None] = None

    @cached_property
    def title(self) -> str:
        return f"mixed data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node, "mixed")

    def get_node(self, node: object) -> object:
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name) if Is.Class(node) else node

    def __init__(self, function: Callable, is_actual=Nothing):
        super().__init__(function)

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
        self.get_node(node)
        name = f'__{("instance", "class")[Is.Class(node)]}_memoized__'

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def call(self, node: object) -> object:
        self.get_node(node)

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

        except AttributeError as e:
            raise AttributeException(e) from e

        return self.__set__(pivot, value)

    def __get__(self, instance: object, klass: object) -> object:
        return self.call(instance or klass)

    def __set__(self, node: object, value: object) -> object:
        cache = self.get_cache(node)

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value

    def __delete__(self, node: object) -> None:
        cache = self.get_cache(node)
        with suppress(KeyError):
            del cache[self.name]
