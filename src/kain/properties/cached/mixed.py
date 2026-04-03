from contextlib import suppress
from functools import cached_property
from typing import ClassVar, override

from kain.internals import Is, Who, get_owner
from kain.properties.cached.klass import (
    class_cached_property,
    class_parent_cached_property,
)
from kain.properties.primitives import ContextFaultError

__all__ = (
    "mixed_cached_property",
    "mixed_parent_cached_property",
)


class mixed_parent_cached_property(class_parent_cached_property):

    @cached_property
    def title(self) -> str:
        return f"mixed data-descriptor {Who.Addr(self)}".strip()

    @override
    def header_with_context(self, node: object) -> str:
        return self.footer(node, "mixed")

    @override
    def get_node(self, node: object) -> object:
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return get_owner(node, self.name) if Is.Class(node) else node

    def get_cache(self, node: object) -> dict[str, object]:
        self.get_node(node)
        name = f'__{("instance", "class")[Is.Class(node)]}_memoized__'

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def __get__(self, instance: object, klass: object) -> object:
        return self.call(instance or klass)


class mixed_cached_property(mixed_parent_cached_property):
    @override
    def get_node(self, node: object) -> object:
        if node is None:
            msg = f"{self.header_with_context(node)}, {node=}"
            raise ContextFaultError(msg)
        return node

    @class_cached_property
    def here(cls) -> type[mixed_parent_cached_property]:
        return mixed_parent_cached_property
