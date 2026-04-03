from asyncio import ensure_future
from functools import cached_property
from inspect import iscoroutinefunction
from typing import ClassVar, override

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


class class_property(BaseProperty):

    klass: ClassVar[bool | None] = True

    @cached_property
    def title(self) -> str:
        return f"class descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node)

    def get_node(self, node: object) -> object:
        if node is None or not Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)

        return get_owner(node, self.name)

    def call(self, node: object) -> object:
        self.get_node(node)
        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                return ensure_future(value)
            return value

        except AttributeError as e:
            raise AttributeException(e) from e

    def __get__(self, instance: object, klass: object) -> object:
        return self.call(klass)


class mixed_property(BaseProperty):

    klass: ClassVar[bool | None] = None

    @cached_property
    def title(self) -> str:
        return f"mixed descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node, "mixed")

    def get_node(self, node: object) -> object:
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name) if Is.Class(node) else node

    def call(self, node: object) -> object:
        self.get_node(node)
        try:
            value = self.function(node)
            if iscoroutinefunction(self.function):
                return ensure_future(value)
            return value

        except AttributeError as e:
            raise AttributeException(e) from e

    def __get__(self, instance: object, klass: object) -> object:
        return self.call(instance or klass)
