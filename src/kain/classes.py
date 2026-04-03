"""Sentinel values and the Singleton metaclass.

This module provides:
    - ``Missing``: a sentinel class whose instances are never equal to
      anything (including themselves) and are always falsy.
    - ``Nothing``: the default global instance of ``Missing``.
    - ``Singleton``: a metaclass that guarantees only one instance of a
      class is ever created.

Example:
    >>> from kain.classes import Nothing, Singleton
    >>> bool(Nothing)
    False
    >>> Nothing == Nothing
    False

    >>> class Database(metaclass=Singleton):
    ...     def __init__(self, url: str) -> None:
    ...         self.url = url
    ...
    >>> db1 = Database("sqlite://")
    >>> db2 = Database("postgres://")
    >>> db1 is db2
    True
    >>> db2.url
    'sqlite://'
"""

from typing import override

from kain.internals import Who

__all__ = "Missing", "Nothing", "Singleton"


class Missing:
    """Sentinel object that is always falsy and never equal to anything."""

    @override
    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        return False

    @override
    def __eq__(self, _: object) -> bool:
        return False

    @override
    def __repr__(self) -> str:
        return f"<{Who.Name(self, addr=True)}>"


Nothing = Missing()
"""Global singleton sentinel used to represent an absent value."""


class Singleton(type):
    """Metaclass that creates at most one instance of a class.

    The first call to the class constructor creates and caches the
    instance. All subsequent calls return the cached instance,
    ignoring any new arguments.
    """

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attributes: dict[str, object],
    ) -> None:
        cls.instance: object | Missing = Nothing
        super().__init__(name, bases, attributes)

    @override
    def __call__(cls, *args: object, **kw: object) -> object:
        if cls.instance is Nothing:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance
