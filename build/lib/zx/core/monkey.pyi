from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import ClassVar, Concatenate, ParamSpec, TypeVar

_P = ParamSpec("_P")
_T = TypeVar("_T")

class Monkey:
    """Namespace for reversible runtime attribute patching utilities."""

    mapping: ClassVar[dict[object, object]]

    @classmethod
    def replace(
        cls,
        target: str | ModuleType | tuple[object, str],
        replacement: _T,
    ) -> _T: ...
    @classmethod
    def bind(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[[Callable[..., _T]], Callable[_P, _T]]: ...
    @classmethod
    def wrap(
        cls,
        node: str | object,
        name: str | None = None,
        decorator: Callable[..., object] | None = None,
    ) -> Callable[
        [Callable[Concatenate[object, _P], _T]],
        Callable[_P, _T],
    ]: ...
