from typing import override

from kain.internals import Is
from kain.properties.cached.klass import class_cached_property
from kain.properties.cached.mixed import (
    mixed_cached_property,
    mixed_parent_cached_property,
)

__all__ = (
    "pre_cached_property",
    "pre_parent_cached_property",
)


class pre_parent_cached_property(mixed_parent_cached_property):

    @override
    def __set__(self, node: object, value: object) -> object:
        cache = self.get_cache(node)

        if not Is.Class(node):
            return value

        if not self.is_actual:
            cache[self.name] = value

        else:
            cache[self.name] = value, self.is_actual(self, node)

        return value


class pre_cached_property(mixed_cached_property):
    @class_cached_property
    def here(cls) -> type[pre_parent_cached_property]:
        return pre_parent_cached_property
