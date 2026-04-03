import atexit
import signal
import sys
import threading
import time
import warnings
from collections.abc import Callable
from datetime import datetime
from functools import cache
from logging import getLogger
from pathlib import Path
from signal import signal as bind
from types import FrameType, TracebackType
from typing import Any, Protocol

from kain.classes import Singleton
from kain.internals import Who

__all__ = (
    "on_quit",
    "quit_at",
)

logger = getLogger(__name__)
NeedRestart: bool = False


class _OnChangeCallable(Protocol):
    def __call__(self, *, sleep: float = 0.0) -> bool: ...
    sleep: Callable[[float, float], bool]


class on_quit(metaclass=Singleton):

    def __init__(self) -> None:
        self.callbacks: list[Callable[[], Any]] = []
        self.hooks_chain: list[
            Callable[
                [type[BaseException], BaseException, TracebackType | None],
                Any,
            ],
        ] = []

        self.original_hook: Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ] = sys.excepthook
        self.already_called: bool = False
        self._proxy = self.exceptions_hooks_proxy

        self.inject_hook()
        self.inject_signal_handler()
        self.inject_threading_hook()

        atexit.register(self.teardown)

    def inject_hook(self) -> None:
        sys.excepthook = self._proxy

    def exceptions_hooks_proxy(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        if sys.excepthook is not self._proxy:
            self.hooks_chain.append(sys.excepthook)
            self.inject_hook()

        for hook in (*self.hooks_chain, self.original_hook):
            try:
                hook(exc_type, exc_value, traceback)
            except Exception as e:  # noqa: BLE001
                warnings.warn(
                    f"{Who.Is(hook)}: {e!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self.teardown()

    def inject_signal_handler(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
            bind(sig, self.signal_handler)

    def signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        self.teardown()
        sys.exit(1)

    def inject_threading_hook(self) -> None:
        threading.excepthook = self.threading_handler

    def threading_handler(self, args: threading.ExceptHookArgs) -> None:
        if args.exc_type is None or args.exc_type is SystemExit:
            return

        self._proxy(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )

    def restore_original_handlers(self) -> None:
        bind(signal.SIGINT, signal.SIG_DFL)
        bind(signal.SIGTERM, signal.SIG_DFL)
        bind(signal.SIGQUIT, signal.SIG_DFL)

        sys.excepthook = self.original_hook
        threading.excepthook = threading.__excepthook__

    def schedule(self, func: Callable[[], Any]) -> None:
        self.callbacks.append(func)

    def add_hook(
        self,
        func: Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ],
    ) -> None:
        self.hooks_chain.append(func)

    def teardown(self) -> None:
        if self.already_called:
            return

        try:
            for func in self.callbacks:
                try:
                    func()
                except BaseException as e:  # noqa: BLE001
                    warnings.warn(
                        f"{Who.Is(func)}: {e!r}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        finally:
            self.already_called = True
            self.restore_original_handlers()


@cache
def get_selfpath() -> Path:
    return Path(sys.argv[0]).resolve()


def get_mtime() -> float:
    return get_selfpath().stat().st_mtime


@cache
def quit_at(
    *,
    func: Callable[..., Any] = sys.exit,
    signal: int = 0,
    errno: int = 137,
    **kw: Any,
) -> _OnChangeCallable:
    """Quit the program when the runned file is updated or signal is received."""

    def handler(*_: Any) -> None:
        global NeedRestart  # noqa: PLW0603
        NeedRestart = True
        logger.warning(f"{signal=} received")

    if signal:
        bind(signal, handler)

    initial_stamp = get_mtime()

    def on_change(*, sleep: float = 0.0) -> bool:
        if NeedRestart and signal:
            logger.warning(f"stop by {signal=}")
            func(errno)
            return False

        try:
            if initial_stamp != (ctime := get_mtime()):
                file = str(get_selfpath())
                when = datetime.utcfromtimestamp(ctime)
                logger.warning(
                    f"{file=} updated at {when} "
                    f"({time.time() - ctime:.2f}s ago), stop",
                )
                func(errno)
                return False

        except FileNotFoundError:
            logger.warning(f"{get_selfpath()} removed? stop")
            return False

        if sleep := (sleep or kw.get("sleep", 0.0)):
            time.sleep(sleep)

        return True

    def sleep(wait: float = 0.0, /, poll: float = 0.0) -> bool:
        if not wait:
            return True

        poll = poll or kw.get("poll", 2.5)
        deadline = time.time() + wait

        while (solution := on_change()) and time.time() < deadline:
            time.sleep(poll)
        return solution

    on_change.sleep = sleep  # type: ignore[attr-defined]
    return on_change  # type: ignore[return-value]
