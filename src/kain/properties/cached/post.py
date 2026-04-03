from typing import override

from kain.internals import Is
from kain.properties.cached.klass import class_cached_property
from kain.properties.cached.mixed import (
    mixed_cached_property,
    mixed_parent_cached_property,
)

__all__ = (
    "post_cached_property",
    "post_parent_cached_property",
)


class post_parent_cached_property(mixed_parent_cached_property):

    @override
    def __set__(self, node: object, value: object) -> object:
        self.get_node(node)
        if Is.Class(node):
            return value
        return super().__set__(node, value)


class post_cached_property(mixed_cached_property):

    @override
    def __set__(self, node: object, value: object) -> object:
        self.get_node(node)
        if Is.Class(node):
            return value
        return super().__set__(node, value)

    @class_cached_property
    def here(cls) -> type[post_parent_cached_property]:
        return post_parent_cached_property
