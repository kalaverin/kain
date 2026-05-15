from threading import RLock
from typing import Any, override

from kain import Who

__all__ = ("Missing", "Nothing", "Singleton")


class Missing:
    """Sentinel object that is always falsy and never equal to anything."""

    __slots__: tuple[Any, ...] = ()

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


Nothing: Missing = Missing()


class Singleton(type):
    """Metaclass that creates at most one instance of a class.

    The first call to the class constructor creates and caches the
    instance. All subsequent calls return the cached instance,
    ignoring any new arguments.
    """

    _lock: RLock = RLock()

    def __init__(
        cls,
        name: str,
        bases: tuple[type, ...],
        attributes: dict[str, object],
    ) -> None:
        """Initialize singleton state."""
        cls.instance: object | Missing = Nothing
        super().__init__(name, bases, attributes)

    @override
    def __call__(cls, *args: object, **kw: object) -> object:
        """Return the cached instance, creating it if necessary."""

        if cls.instance is Nothing:
            with cls._lock:
                if cls.instance is Nothing:
                    cls.instance = super().__call__(*args, **kw)
        return cls.instance
