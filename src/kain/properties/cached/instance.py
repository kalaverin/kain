from contextlib import suppress
from functools import cached_property as base_cached_property

from kain.internals import Is, Who
from kain.properties.cached.klass import (
    class_parent_cached_property,
)
from kain.properties.primitives import (
    ContextFaultError,
)

__all__ = (
    "cached_property",
)


class cached_property(class_parent_cached_property):

    @base_cached_property
    def title(self) -> str:
        return f"instance data-descriptor {Who.Addr(self)}".strip()

    def get_node(self, node: object) -> object:
        if node is None or Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            if node is None and not self.klass:
                msg = f"{msg}; looks like as non-instance invokation"
            raise ContextFaultError(msg)
        return node

    def get_cache(self, node: object) -> dict[str, object]:
        self.get_node(node)
        name = "__instance_memoized__"

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def __get__(self, instance: object, klass: object) -> object:
        if instance is None:
            raise ContextFaultError(self.header_with_context(klass))
        return self.call(instance)
