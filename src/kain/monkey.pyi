from collections.abc import Callable
from types import ModuleType
from typing import Any, ClassVar

__all__ = ("Monkey",)

class Monkey:
    mapping: ClassVar[dict[object, object]]
    @classmethod
    def expect(
        cls,
        *exceptions: type[BaseException],
    ) -> Callable[[Callable[..., object]], Any]: ...
    @classmethod
    def patch(
        cls,
        module: str | ModuleType | tuple[object, str],
        new: object,
    ) -> object: ...
    @classmethod
    def bind(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...
    @classmethod
    def wrap(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., object]], Callable[..., object]]: ...
