"""Signal handling and graceful shutdown utilities.

This module provides mechanisms for:

* **Graceful shutdown on signals** – ``on_quit`` singleton that schedules
callbacks and ensures they run on SIGINT/SIGTERM/SIGQUIT or uncaught exceptions.
* **File-change detection** – ``quit_at`` polls the main script's mtime and
exits when it changes (useful for development auto-reloaders).

Example:
    >>> from kain.signals import on_quit, quit_at
    >>> @on_quit().schedule
    ... def cleanup():
    ...     print("Shutting down...")

    >>> # In a daemon/main loop, check for file changes every 2.5s
    >>> checker = quit_at(signal=signal.SIGUSR1)  # Also exit on SIGUSR1
    >>> while checker.sleep(60):  # Returns False when exiting
    ...     pass
"""

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

#: Global flag set to ``True`` when a signal handler requests a restart.
#: Used by ``quit_at`` to detect external restart requests.
NeedRestart: bool = False


class _OnChangeCallable(Protocol):
    """Protocol for the callable returned by :func:`quit_at`."""

    def __call__(self, *, sleep: float = 0.0) -> bool:
        """Check if restart is needed; optionally sleep before returning."""
        ...

    #: Callable that sleeps while periodically checking for changes.
    sleep: Callable[[float, float], bool]


class on_quit(metaclass=Singleton):
    """Singleton orchestrator for graceful application shutdown.

    ``on_quit`` registers itself with ``atexit``, replaces ``sys.excepthook``,
    ``threading.excepthook``, and signal handlers for SIGINT/SIGTERM/SIGQUIT.
    When any of these triggers, all scheduled callbacks are executed exactly
    once and the original handlers are restored.

    The singleton pattern ensures that multiple instantiations or imports
    within the same process all refer to the same state.
    """

    def __init__(self) -> None:
        """Initialize the singleton state.

        Creates internal storage for callbacks and hooks, saves the original
        exception hooks, and registers with ``atexit`` and signal handlers.
        """
        #: List of no-argument functions to call during teardown.
        self.callbacks: list[Callable[[], Any]] = []

        #: List of exception hook functions with signature
        #: ``(exc_type, exc_value, traceback) -> Any``.
        self.hooks_chain: list[
            Callable[
                [type[BaseException], BaseException, TracebackType | None],
                Any,
            ],
        ] = []

        #: Saved reference to the original ``sys.excepthook``.
        self.original_hook: Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ] = sys.excepthook

        #: Guard to ensure teardown runs only once.
        self.already_called: bool = False

        #: Bound method reference for use as ``sys.excepthook`` replacement.
        self._proxy = self.exceptions_hooks_proxy

        self.inject_hook()
        self.inject_signal_handler()
        self.inject_threading_hook()

        # Ensure cleanup runs at normal interpreter exit.
        atexit.register(self.teardown)

    def inject_hook(self) -> None:
        """Replace ``sys.excepthook`` with our proxy."""
        sys.excepthook = self._proxy

    def exceptions_hooks_proxy(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        traceback: TracebackType | None,
    ) -> None:
        """Intercept uncaught exceptions, chain hooks, then teardown.

        If something else (e.g., a debugger) replaces ``sys.excepthook``,
        we capture that hook in ``hooks_chain`` and re-inject ourselves.
        All hooks in the chain are called in order before delegating to
        the original hook and finally running teardown.
        """
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
        """Register ``signal_handler`` for SIGINT, SIGTERM, and SIGQUIT."""
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT):
            bind(sig, self.signal_handler)

    def signal_handler(self, _signum: int, _frame: FrameType | None) -> None:
        """Handle process signals by tearing down and exiting.

        The exit code is hard-coded to ``1`` to indicate abnormal termination.
        """
        self.teardown()
        sys.exit(1)

    def inject_threading_hook(self) -> None:
        """Replace ``threading.excepthook`` with our proxy."""
        threading.excepthook = self.threading_handler

    def threading_handler(self, args: threading.ExceptHookArgs) -> None:
        """Filter out ``SystemExit`` and ``None`` before delegating.

        The ``threading`` module's hook receives a struct with ``exc_type``,
        ``exc_value``, ``exc_traceback``, and ``thread``. We skip cases where
        the thread died with ``SystemExit`` or no exception at all.
        """
        if args.exc_type is None or args.exc_type is SystemExit:
            return

        self._proxy(
            args.exc_type,
            args.exc_value,
            args.exc_traceback,
        )

    def restore_original_handlers(self) -> None:
        """Revert signal handlers and exception hooks to their original values.

        Called automatically during teardown to avoid interfering with
        subsequent cleanup code or subprocess spawning.
        """
        bind(signal.SIGINT, signal.SIG_DFL)
        bind(signal.SIGTERM, signal.SIG_DFL)
        bind(signal.SIGQUIT, signal.SIG_DFL)

        sys.excepthook = self.original_hook
        threading.excepthook = threading.__excepthook__

    def schedule(self, func: Callable[[], Any]) -> None:
        """Register a callback to be executed during teardown.

        Callbacks are invoked in registration order. Exceptions raised by
        callbacks are caught and emitted as warnings; they do not prevent
        subsequent callbacks from running.
        """
        self.callbacks.append(func)

    def add_hook(
        self,
        func: Callable[
            [type[BaseException], BaseException, TracebackType | None],
            Any,
        ],
    ) -> None:
        """Add a custom exception hook to the chain.

        Hooks are called before the original ``sys.excepthook`` during
        the exception handling flow.
        """
        self.hooks_chain.append(func)

    def teardown(self) -> None:
        """Execute all scheduled callbacks and restore original handlers.

        This method is idempotent; subsequent calls have no effect. It is
        registered with ``atexit`` and also invoked by signal handlers and
        the exception hook proxy.
        """
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
    """Return the resolved Path of ``sys.argv[0]`` (the running script)."""
    return Path(sys.argv[0]).resolve()


def get_mtime() -> float:
    """Return the mtime (modification timestamp) of the running script."""
    return get_selfpath().stat().st_mtime


@cache
def quit_at(
    *,
    func: Callable[..., Any] = sys.exit,
    signal: int = 0,
    errno: int = 137,
    **kw: Any,
) -> _OnChangeCallable:
    """Create a change-detector that exits when the script file is updated.

    The returned callable (and its attached ``sleep`` method) can be used
    in a main loop to periodically check for modifications. If the file's
    mtime changes, ``func(errno)`` is called.

    Optionally, a POSIX signal number can be provided. When that signal is
    received, the global ``NeedRestart`` flag is set to ``True`` and the
    next check will trigger the exit.

    Parameters
    ----------
    func:
        Callable to invoke when a change is detected. Defaults to
        ``sys.exit``.
    signal:
        POSIX signal number to listen for (e.g., ``signal.SIGUSR1``).
        Pass ``0`` to disable signal handling.
    errno:
        Exit code passed to ``func`` on change detection. Default ``137``
        (128 + ``SIGKILL``) conventionally means "external termination".
    **kw:
        Additional keyword arguments merged into the returned callable's
        namespace (used internally for ``sleep`` and ``poll`` defaults).

    Returns
    -------
    _OnChangeCallable
        A callable with signature ``(sleep=0.0) -> bool``. Returns ``False``
        when the application should exit. The callable has a ``sleep``
        method for blocking checks.

    Example:
        >>> import signal
        >>> checker = quit_at(signal=signal.SIGUSR1)
        >>> while checker.sleep(60):  # Block up to 60s
        ...     pass  # Main loop work here
    """

    def handler(*_: Any) -> None:
        """Signal handler that sets the global restart flag."""
        global NeedRestart  # noqa: PLW0603
        NeedRestart = True
        logger.warning(f"{signal=} received")

    if signal:
        bind(signal, handler)

    # Snapshot the mtime at construction time; future comparisons use this.
    initial_stamp = get_mtime()

    def on_change(*, sleep: float = 0.0) -> bool:
        """Check if the file has changed or a restart was requested.

        Returns ``True`` if the application should continue running,
        ``False`` if it should exit (after calling ``func(errno)``).
        """
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
        """Block for up to ``wait`` seconds while polling for changes.

        Parameters
        ----------
        wait:
            Maximum time to block (seconds). Pass ``0`` to return immediately.
        poll:
            Interval between checks (seconds). Defaults to ``2.5`` or the
            value passed in ``kw["poll"]``.

        Returns
        -------
        bool
            ``True`` if the loop should continue, ``False`` if a change was
            detected and ``func(errno)`` was called.
        """
        if not wait:
            return True

        poll = poll or kw.get("poll", 2.5)
        deadline = time.time() + wait

        while (solution := on_change()) and time.time() < deadline:
            time.sleep(poll)
        return solution

    # Attach the sleep method to the returned callable so users can write:
    #   while checker.sleep(60): ...
    on_change.sleep = sleep  # type: ignore[attr-defined]
    return on_change  # type: ignore[return-value]
