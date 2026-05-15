from __future__ import annotations

from typing import Any, override

from kain.properties.cached.klass import class_cached_property
from kain.properties.cached.mixed import (
    mixed_cached_property,
    mixed_parent_cached_property,
)

__all__ = ("post_cached_property", "post_parent_cached_property")
T_co: Any

class post_parent_cached_property[T_co](
    mixed_parent_cached_property[T_co],
):
    @override
    def __set__(self, node: Any, value: Any) -> Any: ...

class post_cached_property[T_co](
    mixed_cached_property[T_co],
):
    @override
    def __set__(self, node: Any, value: Any) -> Any: ...
    @class_cached_property
    @override
    def here(  # pyrefly: ignore[missing-override-decorator]
        cls,
    ) -> type[post_parent_cached_property[Any]]: ...
