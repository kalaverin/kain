"""Tests for kain.signals module."""

from __future__ import annotations

import signal
import sys
import threading
import warnings
from types import TracebackType
from typing import Any
from unittest.mock import ANY, MagicMock, patch

import pytest

from kain.classes import Nothing
from kain.signals import (
    NeedRestart,
    get_mtime,
    get_selfpath,
    on_quit,
    quit_at,
)


@pytest.fixture(autouse=True)
def _isolate_on_quit() -> None:
    """Restore global state and reset the on_quit singleton after each test."""
    # Ensure a clean baseline before each test
    sys.excepthook = sys.__excepthook__
    original_excepthook = sys.excepthook
    original_threading_hook = threading.excepthook

    # Reset singleton state so the next test gets a fresh instance
    on_quit.instance = Nothing  # type: ignore[attr-defined]

    yield

    # Tear down any created instance and restore hooks
    inst = on_quit.instance
    if inst is not Nothing and hasattr(inst, "restore_original_handlers"):
        inst.restore_original_handlers()

    on_quit.instance = Nothing  # type: ignore[attr-defined]
    sys.excepthook = original_excepthook
    threading.excepthook = original_threading_hook


def _fresh_instance() -> Any:
    """Create a fresh on_quit singleton instance."""
    on_quit.instance = Nothing  # type: ignore[attr-defined]
    return on_quit()


def _make_except_hook_args(
    exc_type: type[BaseException] | None,
    exc_value: BaseException | None,
    exc_traceback: TracebackType | None,
    thread: threading.Thread | None = None,
) -> threading.ExceptHookArgs:
    """Build threading.ExceptHookArgs from positional tuple (structseq)."""
    if thread is None:
        thread = threading.current_thread()
    return threading.ExceptHookArgs(
        (exc_type, exc_value, exc_traceback, thread),
    )


class TestOnSystemExit:
    def test_on_quit_is_singleton(self) -> None:
        inst1 = _fresh_instance()
        inst2 = on_quit()
        assert inst1 is inst2

    def test_schedule_calls_callback_on_teardown(self) -> None:
        callback = MagicMock()
        inst = _fresh_instance()
        inst.schedule(callback)
        inst.teardown()
        callback.assert_called_once_with()

    def test_teardown_is_idempotent(self) -> None:
        callback = MagicMock()
        inst = _fresh_instance()
        inst.schedule(callback)
        inst.teardown()
        inst.teardown()
        assert callback.call_count == 1

    def test_teardown_catches_callback_exceptions(self) -> None:
        callback = MagicMock(side_effect=RuntimeError("boom"))
        inst = _fresh_instance()
        inst.schedule(callback)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst.teardown()
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "boom" in str(w[0].message)

    def test_threading_handler_skips_system_exit(self) -> None:
        inst = _fresh_instance()
        args = _make_except_hook_args(SystemExit, SystemExit(1), None)
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_not_called()

    def test_threading_handler_skips_none_exc_type(self) -> None:
        inst = _fresh_instance()
        args = _make_except_hook_args(None, None, None)
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_not_called()

    def test_threading_handler_proxies_other_exceptions(self) -> None:
        inst = _fresh_instance()
        tb: TracebackType | None = None
        try:
            raise ValueError("test")
        except ValueError:
            tb = sys.exc_info()[2]
        args = _make_except_hook_args(ValueError, ValueError("test"), tb)
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_called_once_with(
                ValueError,
                args.exc_value,
                tb,
            )

    def test_restore_original_handlers_resets_sigquit(self) -> None:
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.restore_original_handlers()
        calls = [c[0][0] for c in mock_bind.call_args_list]
        assert signal.SIGINT in calls
        assert signal.SIGTERM in calls
        assert signal.SIGQUIT in calls
        assert signal.SIGHUP not in calls

    def test_exceptions_hooks_proxy_calls_hooks_and_teardown(self) -> None:
        inst = _fresh_instance()
        hook_calls: list[tuple[Any, ...]] = []
        original_calls: list[tuple[Any, ...]] = []

        def hook(*args: Any) -> None:
            hook_calls.append(args)

        def original_hook(*args: Any) -> None:
            original_calls.append(args)

        inst.add_hook(hook)
        inst.original_hook = original_hook
        inst.exceptions_hooks_proxy(
            RuntimeError,
            RuntimeError("boom"),
            None,
        )
        assert len(hook_calls) == 1
        assert hook_calls[0][0] is RuntimeError
        assert str(hook_calls[0][1]) == "boom"
        assert hook_calls[0][2] is None
        assert len(original_calls) == 1
        assert original_calls[0][0] is RuntimeError
        assert str(original_calls[0][1]) == "boom"
        assert original_calls[0][2] is None
        assert inst.already_called is True

    def test_signal_handler_calls_teardown_and_exits(self) -> None:
        inst = _fresh_instance()
        with patch.object(inst, "teardown") as mock_teardown:
            with pytest.raises(SystemExit) as exc_info:
                inst.signal_handler(15, None)
            assert exc_info.value.code == 1
            mock_teardown.assert_called_once()

    def test_inject_hook_replaces_excepthook(self) -> None:
        original = sys.excepthook
        inst = _fresh_instance()
        assert sys.excepthook is inst._proxy
        sys.excepthook = original

    def test_teardown_restores_handlers(self) -> None:
        original_excepthook = sys.excepthook
        inst = _fresh_instance()
        inst.teardown()
        assert sys.excepthook is original_excepthook
        assert threading.excepthook is threading.__excepthook__


class TestQuitAt:
    def test_get_selfpath_returns_absolute_path(self) -> None:
        path = get_selfpath()
        assert path.is_absolute()

    def test_get_mtime_returns_float(self) -> None:
        mtime = get_mtime()
        assert isinstance(mtime, float)
        assert mtime > 0.0

    def test_quit_at_returns_callable_with_sleep_attr(self) -> None:
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at()
        assert callable(checker)
        assert hasattr(checker, "sleep")
        assert callable(checker.sleep)

    def test_quit_at_on_change_true_when_no_changes(self) -> None:
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at()
        assert checker() is True

    def test_quit_at_sleep_method_with_zero_wait(self) -> None:
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at()
        assert checker.sleep(0.0) is True
        assert checker.sleep(0.0, poll=0.1) is True

    def test_quit_at_registers_signal_handler(self) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        quit_at.cache_clear()  # type: ignore[attr-defined]
        with patch("kain.signals.bind") as mock_bind:
            checker = quit_at(func=mock_exit, signal=signal.SIGUSR1)
            mock_bind.assert_called_once_with(signal.SIGUSR1, ANY)
        handler = mock_bind.call_args[0][1]
        assert sig_mod.NeedRestart is False
        handler(1, None)
        assert sig_mod.NeedRestart is True
        sig_mod.NeedRestart = False

    def test_quit_at_on_change_triggers_when_needrestart_set(self) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at(func=mock_exit, signal=signal.SIGUSR1)
        sig_mod.NeedRestart = True
        try:
            assert checker() is False
            mock_exit.assert_called_once_with(137)
        finally:
            sig_mod.NeedRestart = False

    def test_quit_at_on_change_file_not_found(self) -> None:
        mock_exit = MagicMock()
        quit_at.cache_clear()  # type: ignore[attr-defined]
        with patch("kain.signals.get_mtime") as mock_mtime:
            mock_mtime.return_value = 1.0
            checker = quit_at(func=mock_exit)
            mock_mtime.side_effect = FileNotFoundError
            assert checker() is False

    def test_quit_at_sleep_with_short_wait(self) -> None:
        mock_exit = MagicMock()
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at(func=mock_exit)
        result = checker.sleep(0.01, poll=0.001)
        assert result is True


class TestModuleAttributes:
    def test_needrestart_is_bool(self) -> None:
        assert isinstance(NeedRestart, bool)

    def test_all_exports_exist(self) -> None:
        import kain.signals as mod

        for name in mod.__all__:
            assert hasattr(mod, name)
