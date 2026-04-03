"""Cached property descriptors.

This sub-package provides caching variants of the basic descriptors defined
in ``kain.properties.class_property`` and ``kain.properties.primitives``.

Each descriptor family comes in two flavours:

* **Plain** (e.g. ``class_cached_property``) – caches directly on the
  ``node`` that was passed to ``__get__``.
* **Parent-aware** (e.g. ``class_parent_cached_property``) – caches on the
  *owning* class in the MRO (found via :func:`kain.internals.get_owner`).

The mixed descriptors additionally branch their cache storage depending on
whether they are accessed on an instance or on a class.
"""

from kain.properties.cached.instance import (
    cached_property,
)
from kain.properties.cached.klass import (
    class_cached_property,
    class_parent_cached_property,
)
from kain.properties.cached.mixed import (
    mixed_cached_property,
    mixed_parent_cached_property,
)
from kain.properties.cached.post import (
    post_cached_property,
    post_parent_cached_property,
)
from kain.properties.cached.pre import (
    pre_cached_property,
    pre_parent_cached_property,
)

__all__ = (
    "cached_property",
    "class_cached_property",
    "class_parent_cached_property",
    "mixed_cached_property",
    "mixed_parent_cached_property",
    "post_cached_property",
    "post_parent_cached_property",
    "pre_cached_property",
    "pre_parent_cached_property",
)
