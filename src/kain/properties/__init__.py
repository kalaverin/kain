"""Public API for the ``kain.properties`` package.

This module aggregates all property-descriptor implementations so they can be
imported from a single namespace.  The primary user-facing entry point is the
:class:`pin` decorator namespace — a family of replacements for
:func:`property` and :func:`functools.cached_property` that support
instance-only, class-only, mixed instance/class, parent-aware, and TTL-aware
caching semantics.
"""

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
    AttributeException,
    BaseProperty,
    ContextFaultError,
    PropertyError,
    ReadOnlyError,
    bound_property,
)
from kain.properties.proxy_to import (
    proxy_to,
)

__all__ = (
    "AttributeException",
    "BaseProperty",
    "ContextFaultError",
    "PropertyError",
    "ReadOnlyError",
    "bound_property",
    "cached_property",
    "class_cached_property",
    "class_parent_cached_property",
    "class_property",
    "mixed_cached_property",
    "mixed_parent_cached_property",
    "mixed_property",
    "pin",
    "post_cached_property",
    "post_parent_cached_property",
    "pre_cached_property",
    "pre_parent_cached_property",
    "proxy_to",
)


class pin(bound_property):
    """Namespace decorator for the ``kain`` property-descriptor family.

    ``pin`` itself behaves like :class:`bound_property` (an instance-only,
    non-caching descriptor), but it exposes class attributes that point to the
    more specialized implementations:

    * ``pin.native``      → :class:`cached_property` (instance-level cache)
    * ``pin.cls``         → :class:`class_cached_property` (class-level cache)
    * ``pin.any``         → :class:`mixed_cached_property` (mixed cache)
    * ``pin.pre``         → :class:`pre_cached_property` (cache on class only)
    * ``pin.post``        → :class:`post_cached_property` (cache on instance only)

    In addition, each ``*_cached_property`` has a ``*_parent_cached_property``
    variant (accessible via ``.with_parent`` on the base classes) that resolves
    the descriptor against the *owning* class in the MRO rather than the
    concrete class on which the attribute was accessed.
    """

    # Mapping from short alias to the concrete descriptor class.
    # These are used so end-users can write ``@pin.native`` instead of
    # ``@cached_property``.
    native = cached_property
    cls = class_cached_property
    any = mixed_cached_property
    pre = pre_cached_property
    post = post_cached_property
