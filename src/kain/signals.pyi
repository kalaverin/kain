from collections.abc import Callable
from pathlib import Path
from types import FrameType, TracebackType
from typing import Any, Protocol

from kain.classes import Singleton

__all__ = ("on_quit", "quit_at")

NeedRestart: bool

class _OnChangeCallable(Protocol):
    def __call__(self, *, sleep: float = 0.0) -> bool: ...
    sleep: Callable[[float, float], bool]

class on_quit(metaclass=Singleton):
    callbacks: list[Callable[[], Any]]
    hooks_chain: list[
        Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ],
    ]
    original_hook: Callable[
        [type[BaseException], BaseException, TracebackType | None],
        Any,
    ]
    already_called: bool
    _proxy: Callable[..., Any]
    def __init__(self) -> None: ...
    def inject_hook(self) -> None: ...
    def exceptions_hooks_proxy(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType | None,
    ) -> None: ...
    def inject_signal_handler(self) -> None: ...
    def signal_handler(
        self,
        _signum: int,
        _frame: FrameType | None,
    ) -> None: ...
    def inject_threading_hook(self) -> None: ...
    def threading_handler(self, args: Any) -> None: ...
    def restore_original_handlers(self) -> None: ...
    def schedule(self, func: Callable[[], Any]) -> None: ...
    def add_hook(
        self,
        func: Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ],
    ) -> None: ...
    def teardown(self) -> None: ...

def get_selfpath() -> Path: ...
def get_mtime() -> float: ...
def quit_at(
    *,
    func: Callable[..., Any] = ...,
    signal: int = 0,
    errno: int = 137,
    **kw: Any,
) -> _OnChangeCallable: ...
