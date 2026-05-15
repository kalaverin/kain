from typing import Any, override

__all__ = ("Missing", "Nothing", "Singleton")

class Missing:
    @override
    def __hash__(self) -> int: ...
    def __bool__(self) -> bool: ...
    @override
    def __eq__(self, _: object) -> bool: ...
    @override
    def __repr__(self) -> str: ...

class Singleton(type):
    _lock: Any
    def __init__(
        cls: type[Any],
        name: str,
        bases: tuple[type, ...],
        attributes: dict[str, object],
    ) -> None: ...
    @override
    def __call__(cls, *args: object, **kw: object) -> object: ...

Nothing: Missing
