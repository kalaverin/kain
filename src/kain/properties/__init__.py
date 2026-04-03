from kain.properties.cached import (
    cached_property,
    class_cached_property,
    class_parent_cached_property,
    mixed_cached_property,
    mixed_parent_cached_property,
    post_cached_property,
    post_parent_cached_property,
    pre_cached_property,
    pre_parent_cached_property,
)
from kain.properties.class_property import (
    class_property,
    mixed_property,
)
from kain.properties.primitives import (
    bound_property,
)

__all__ = (
    "bound_property",
    "cached_property",
    "class_cached_property",
    "class_parent_cached_property",
    "class_property",
    "mixed_cached_property",
    "mixed_parent_cached_property",
    "mixed_property",
    "post_cached_property",
    "post_parent_cached_property",
    "pre_cached_property",
    "pre_parent_cached_property",
)


class pin(bound_property):
    native = cached_property
    cls = class_property
    any = mixed_property
    pre = pre_cached_property
    post = post_cached_property
