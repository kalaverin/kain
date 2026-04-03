"""Instance-level cached property descriptor.

This module defines :class:`cached_property` — the caching equivalent of
:class:`kain.properties.primitives.bound_property`.  It stores computed
values in a per-instance dictionary named ``__instance_memoized__`` and
supports parent-aware resolution (via inheritance from
:class:`class_parent_cached_property`).
"""

from contextlib import suppress
from functools import cached_property as base_cached_property

from kain.internals import Is, Who
from kain.properties.cached.klass import (
    class_parent_cached_property,
)
from kain.properties.primitives import (
    ContextFaultError,
)

__all__ = ("cached_property",)


class cached_property(class_parent_cached_property):
    """Instance-level cached descriptor.

    When accessed on an instance, the underlying function is called once,
    the result is stored in ``instance.__dict__["__instance_memoized__"]``,
    and subsequent accesses return the cached value.

    Access on the class itself (``instance is None``) raises
    :exc:`ContextFaultError`.

    Parent-aware resolution
    -----------------------
    Because this class inherits from ``class_parent_cached_property``,
    ``with_parent`` works out of the box: the parent descriptor up the MRO
    is located, evaluated, and its result is injected as the second argument
    to the overriding function.
    """

    @base_cached_property
    def title(self) -> str:
        return f"instance data-descriptor {Who.Addr(self)}".strip()

    def get_node(self, node: object) -> object:
        """Assert ``node`` is an instance (not a class) and return it."""
        if node is None or Is.Class(node):
            msg = f"{self.header_with_context(node)}, {node=}"
            if node is None and not self.klass:
                msg = f"{msg}; looks like as non-instance invokation"
            raise ContextFaultError(msg)
        return node

    def get_cache(self, node: object) -> dict[str, object]:
        """Return (creating if necessary) the ``__instance_memoized__`` dict.

        The cache lives directly on ``node`` (the instance), so every
        instance has isolated memoization.
        """
        self.get_node(node)
        name = "__instance_memoized__"

        if hasattr(node, "__dict__"):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    def __get__(self, instance: object, klass: object) -> object:
        """Descriptor protocol hook — only instance access is allowed."""
        if instance is None:
            raise ContextFaultError(self.header_with_context(klass))
        return self.call(instance)
