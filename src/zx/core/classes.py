from threading import RLock
from typing import Any, TypeVar, cast, override

from zx.core import _who

__all__ = ("Missing", "Nothing", "Singleton")


_SingletonInstance = TypeVar("_SingletonInstance")


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
        return f"<{_who.Name(self, addr=True)}>"


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
    def __call__(
        cls: type[_SingletonInstance],
        *args: object,
        **kw: object,
    ) -> _SingletonInstance:
        """Return the cached instance, creating it if necessary."""

        singleton = cast("Singleton", cls)
        if singleton.instance is Nothing:
            with singleton._lock:
                if singleton.instance is Nothing:
                    singleton.instance = super().__call__(*args, **kw)
        return cast("_SingletonInstance", singleton.instance)
