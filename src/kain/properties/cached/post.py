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
        cache = self.get_cache(node)

        if Is.Class(node):
            return value

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value


class post_cached_property(mixed_cached_property):
    @class_cached_property
    def here(cls) -> type[post_parent_cached_property]:
        return post_parent_cached_property
