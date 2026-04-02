from typing import Any, override

from kain.internals import Who

__all__ = "Missing", "Nothing", "Singleton"


class Missing:

    @override
    def __hash__(self) -> int:
        return id(self)

    def __bool__(self) -> bool:
        return False

    @override
    def __eq__(self, _) -> bool:
        return False

    @override
    def __repr__(self) -> str:
        return f"<{Who.Name(self, addr=True)}>"


Nothing = Missing()


class Singleton(type):

    def __init__(
        cls,
        name: str,
        parents: tuple[()] | tuple[type] | tuple[type, ...],
        attributes: dict[str, Any],
    ) -> None:
        cls.instance: Any | Missing = Nothing
        super().__init__(name, parents, attributes)

    @override
    def __call__(cls, *args: Any, **kw: dict[str, Any]) -> Any:
        if cls.instance is not Missing:
            cls.instance = super().__call__(*args, **kw)
        return cls.instance
