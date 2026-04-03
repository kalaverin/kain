"""Tests for kain.signals module."""

from __future__ import annotations

import signal
import sys
import threading
import warnings
from functools import partial
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


# ============================================================================
# Expanded tests below
# ============================================================================


class TestOnQuitScheduleVariations:
    """Tests for scheduling various callable types."""

    def test_schedule_plain_function(self) -> None:
        calls: list[int] = []
        inst = _fresh_instance()
        inst.schedule(lambda: calls.append(1))
        inst.teardown()
        assert calls == [1]

    def test_schedule_lambda(self) -> None:
        flag = []
        inst = _fresh_instance()
        inst.schedule(lambda: flag.append("ok"))
        inst.teardown()
        assert flag == ["ok"]

    def test_schedule_bound_method(self) -> None:
        class C:
            def __init__(self) -> None:
                self.val = 0

            def incr(self) -> None:
                self.val += 1

        obj = C()
        inst = _fresh_instance()
        inst.schedule(obj.incr)
        inst.teardown()
        assert obj.val == 1

    def test_schedule_callable_class_instance(self) -> None:
        class Counter:
            def __init__(self) -> None:
                self.count = 0

            def __call__(self) -> None:
                self.count += 1

        counter = Counter()
        inst = _fresh_instance()
        inst.schedule(counter)
        inst.teardown()
        assert counter.count == 1

    def test_schedule_functools_partial(self) -> None:
        results: list[int] = []
        inst = _fresh_instance()
        inst.schedule(partial(results.append, 42))
        inst.teardown()
        assert results == [42]

    def test_duplicate_schedule_calls_callback_twice(self) -> None:
        calls: list[int] = []
        inst = _fresh_instance()
        inst.schedule(lambda: calls.append(1))
        inst.schedule(lambda: calls.append(1))
        inst.teardown()
        assert calls == [1, 1]

    def test_schedule_preserves_registration_order(self) -> None:
        order: list[str] = []
        inst = _fresh_instance()
        inst.schedule(lambda: order.append("a"))
        inst.schedule(lambda: order.append("b"))
        inst.schedule(lambda: order.append("c"))
        inst.teardown()
        assert order == ["a", "b", "c"]

    def test_schedule_returns_none(self) -> None:
        inst = _fresh_instance()
        result = inst.schedule(lambda: None)
        assert result is None


class TestOnQuitTeardownExtended:
    """Extended teardown behavior tests."""

    def test_teardown_sets_already_called(self) -> None:
        inst = _fresh_instance()
        assert inst.already_called is False
        inst.teardown()
        assert inst.already_called is True

    def test_teardown_with_empty_callbacks_ok(self) -> None:
        inst = _fresh_instance()
        assert inst.teardown() is None
        assert inst.already_called is True

    def test_teardown_catches_baseexception_in_callback(self) -> None:
        inst = _fresh_instance()
        inst.schedule(lambda: (_ for _ in ()).throw(SystemExit("die")))
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst.teardown()
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "die" in str(w[0].message)

    def test_teardown_restores_handlers_even_if_callback_raises(self) -> None:
        original_excepthook = sys.excepthook
        inst = _fresh_instance()
        inst.schedule(lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            inst.teardown()
        assert sys.excepthook is original_excepthook

    def test_teardown_runs_callbacks_in_order(self) -> None:
        order: list[int] = []
        inst = _fresh_instance()
        inst.schedule(lambda: order.append(1))
        inst.schedule(lambda: order.append(2))
        inst.teardown()
        assert order == [1, 2]

    def test_multiple_teardown_calls_no_additional_callbacks(self) -> None:
        counter = MagicMock()
        inst = _fresh_instance()
        inst.schedule(counter)
        inst.teardown()
        inst.teardown()
        inst.teardown()
        assert counter.call_count == 1


class TestOnQuitHooksExtended:
    """Extended tests for exception hooks and proxy behavior."""

    def test_add_hook_appends_to_hooks_chain(self) -> None:
        inst = _fresh_instance()
        inst.add_hook(lambda *_: None)
        assert len(inst.hooks_chain) == 1

    def test_exception_in_hook_warns_but_continues(self) -> None:
        inst = _fresh_instance()
        calls: list[str] = []

        def bad_hook(*_):
            calls.append("bad")
            raise RuntimeError("hook boom")

        def good_hook(*_):
            calls.append("good")

        inst.add_hook(bad_hook)
        inst.add_hook(good_hook)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst.exceptions_hooks_proxy(RuntimeError, RuntimeError("x"), None)
        assert calls == ["bad", "good"]
        assert len(w) == 1
        assert "hook boom" in str(w[0].message)
        assert inst.already_called is True

    def test_multiple_hooks_all_called(self) -> None:
        inst = _fresh_instance()
        count = 0

        def hook(*_):
            nonlocal count
            count += 1

        inst.add_hook(hook)
        inst.add_hook(hook)
        inst.add_hook(hook)
        inst.exceptions_hooks_proxy(RuntimeError, RuntimeError("x"), None)
        assert count == 3

    def test_exceptions_hooks_proxy_reinjects_if_excepthook_changed(
        self,
    ) -> None:
        inst = _fresh_instance()
        other_hook = MagicMock()
        sys.excepthook = other_hook
        inst.exceptions_hooks_proxy(RuntimeError, RuntimeError("x"), None)
        # teardown() restores sys.excepthook, so we verify by side-effects:
        assert other_hook in inst.hooks_chain
        assert inst.already_called is True

    def test_exceptions_hooks_proxy_recursion_safety(self) -> None:
        """Ensure that changing excepthook inside a hook does not recurse."""
        inst = _fresh_instance()
        call_count = 0

        def mutating_hook(*_):
            nonlocal call_count
            call_count += 1
            sys.excepthook = lambda *_: None

        inst.add_hook(mutating_hook)
        # The proxy should run once, append the lambda, and finish without
        # entering an infinite loop because it only checks at the top.
        inst.exceptions_hooks_proxy(RuntimeError, RuntimeError("x"), None)
        assert call_count == 1
        assert inst.already_called is True

    def test_exceptions_hooks_proxy_calls_original_hook(self) -> None:
        inst = _fresh_instance()
        original = MagicMock()
        inst.original_hook = original
        inst.exceptions_hooks_proxy(ValueError, ValueError("v"), None)
        assert original.call_count == 1
        args = original.call_args[0]
        assert args[0] is ValueError
        assert str(args[1]) == "v"
        assert args[2] is None

    def test_inject_hook_replaces_excepthook_explicitly(self) -> None:
        inst = _fresh_instance()
        sys.excepthook = sys.__excepthook__
        inst.inject_hook()
        assert sys.excepthook is inst._proxy


class TestOnQuitThreadingExtended:
    """Extended threading exception handler tests."""

    @pytest.mark.parametrize(
        "exc_type,should_proxy",
        [
            (SystemExit, False),
            (None, False),
            (ValueError, True),
            (RuntimeError, True),
            (TypeError, True),
            (KeyboardInterrupt, True),
        ],
    )
    def test_threading_handler_various_exceptions(
        self,
        exc_type: type[BaseException] | None,
        should_proxy: bool,
    ) -> None:
        inst = _fresh_instance()
        args = _make_except_hook_args(
            exc_type,
            RuntimeError("x") if exc_type else None,
            None,
        )
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            if should_proxy:
                proxy.assert_called_once()
            else:
                proxy.assert_not_called()

    def test_threading_handler_with_custom_thread(self) -> None:
        inst = _fresh_instance()
        custom_thread = threading.Thread(target=lambda: None, name="custom")
        tb: TracebackType | None = None
        try:
            raise ValueError("t")
        except ValueError:
            tb = sys.exc_info()[2]
        args = _make_except_hook_args(
            ValueError,
            ValueError("t"),
            tb,
            thread=custom_thread,
        )
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_called_once_with(
                ValueError,
                args.exc_value,
                tb,
            )

    def test_threading_handler_with_baseexception(self) -> None:
        inst = _fresh_instance()
        args = _make_except_hook_args(
            ArithmeticError,
            ArithmeticError("a"),
            None,
        )
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_called_once()

    def test_inject_threading_hook_replaces_hook(self) -> None:
        inst = _fresh_instance()
        threading.excepthook = threading.__excepthook__
        inst.inject_threading_hook()
        assert (
            getattr(threading.excepthook, "__func__", None)
            is inst.threading_handler.__func__
        )


class TestOnQuitSignalExtended:
    """Extended signal handler tests."""

    @pytest.mark.parametrize(
        "sig",
        [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT],
    )
    def test_signal_handler_exits_with_code_one(self, sig: int) -> None:
        inst = _fresh_instance()
        with patch.object(inst, "teardown") as mock_teardown:
            with pytest.raises(SystemExit) as exc_info:
                inst.signal_handler(sig, None)
            assert exc_info.value.code == 1
            mock_teardown.assert_called_once()

    @pytest.mark.parametrize(
        "sig",
        [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT],
    )
    def test_inject_signal_handler_binds_signal(self, sig: int) -> None:
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.inject_signal_handler()
        calls = [c[0][0] for c in mock_bind.call_args_list]
        assert sig in calls

    def test_inject_signal_handler_does_not_bind_sighup(self) -> None:
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.inject_signal_handler()
        calls = [c[0][0] for c in mock_bind.call_args_list]
        assert getattr(signal, "SIGHUP", -1) not in calls

    @pytest.mark.parametrize(
        "sig",
        [signal.SIGINT, signal.SIGTERM, signal.SIGQUIT],
    )
    def test_restore_original_handlers_resets_signal(self, sig: int) -> None:
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.restore_original_handlers()
        calls = {c[0][0]: c[0][1] for c in mock_bind.call_args_list}
        assert calls[sig] is signal.SIG_DFL

    def test_restore_original_handlers_does_not_touch_sighup(self) -> None:
        if not hasattr(signal, "SIGHUP"):
            pytest.skip("SIGHUP not available on this platform")
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.restore_original_handlers()
        calls = [c[0][0] for c in mock_bind.call_args_list]
        assert signal.SIGHUP not in calls

    def test_restore_original_handlers_resets_sys_excepthook(self) -> None:
        original = sys.__excepthook__
        inst = _fresh_instance()
        sys.excepthook = lambda *_: None
        inst.restore_original_handlers()
        assert sys.excepthook is original

    def test_restore_original_handlers_resets_threading_excepthook(
        self,
    ) -> None:
        inst = _fresh_instance()
        threading.excepthook = lambda *_: None
        inst.restore_original_handlers()
        assert threading.excepthook is threading.__excepthook__


class TestQuitAtExtended:
    """Extended quit_at tests."""

    @pytest.fixture(autouse=True)
    def _clear_quit_at_cache(self) -> None:
        quit_at.cache_clear()  # type: ignore[attr-defined]

    def test_on_change_detects_mtime_change(self) -> None:
        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit)
        with patch("kain.signals.get_mtime") as mock_mtime:
            mock_mtime.return_value = 999999.0
            assert checker() is False
            mock_exit.assert_called_once_with(137)

    def test_on_change_returns_true_when_stable(self) -> None:
        checker = quit_at()
        assert checker() is True

    def test_on_change_with_sleep_parameter(self) -> None:
        checker = quit_at()
        with patch("kain.signals.time.sleep") as mock_sleep:
            checker(sleep=0.5)
            mock_sleep.assert_called_once_with(0.5)

    def test_on_change_uses_default_sleep_from_kw(self) -> None:
        checker = quit_at(sleep=0.3)
        with patch("kain.signals.time.sleep") as mock_sleep:
            checker()
            mock_sleep.assert_called_once_with(0.3)

    def test_sleep_zero_wait_returns_true(self) -> None:
        checker = quit_at()
        assert checker.sleep(0.0) is True

    def test_sleep_polling_returns_true_when_stable(self) -> None:
        checker = quit_at()
        result = checker.sleep(0.01, poll=0.001)
        assert result is True

    def test_sleep_stops_polling_on_change(self) -> None:
        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit)
        with patch("kain.signals.get_mtime") as mock_mtime:
            # first call inside on_change sees changed mtime
            mock_mtime.return_value = 999999.0
            result = checker.sleep(1.0, poll=0.001)
            assert result is False
            mock_exit.assert_called_once_with(137)

    def test_needrestart_triggers_with_custom_errno(self) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit, signal=signal.SIGUSR1, errno=42)
        sig_mod.NeedRestart = True
        try:
            assert checker() is False
            mock_exit.assert_called_once_with(42)
        finally:
            sig_mod.NeedRestart = False

    def test_needrestart_without_signal_does_nothing(self) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit, signal=0)
        sig_mod.NeedRestart = True
        try:
            # signal=0 means no handler registered, so NeedRestart + signal=0
            # does NOT trigger because condition is `NeedRestart and signal`
            assert checker() is True
            mock_exit.assert_not_called()
        finally:
            sig_mod.NeedRestart = False

    def test_custom_func_return_value_ignored(self) -> None:
        mock_exit = MagicMock(return_value="ignored")
        checker = quit_at(func=mock_exit)
        with patch("kain.signals.get_mtime") as mock_mtime:
            mock_mtime.return_value = 888888.0
            assert checker() is False
            mock_exit.assert_called_once()

    def test_signal_registration_and_handler(self) -> None:
        mock_exit = MagicMock()
        with patch("kain.signals.bind") as mock_bind:
            checker = quit_at(func=mock_exit, signal=signal.SIGUSR2)
            mock_bind.assert_called_once_with(signal.SIGUSR2, ANY)

    def test_signal_unregistration_via_flag_reset(self) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit, signal=signal.SIGUSR1)
        sig_mod.NeedRestart = True
        try:
            checker()
            # After first call, flag is still True but func was called.
            # Actually func(sys.exit) would exit the process; with mock it
            # doesn't reset the flag. We just verify the behavior.
            mock_exit.assert_called_once()
        finally:
            sig_mod.NeedRestart = False

    def test_quit_at_invalid_path_file_not_found(self) -> None:
        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit)
        with patch("kain.signals.get_mtime") as mock_mtime:
            mock_mtime.side_effect = FileNotFoundError
            assert checker() is False

    def test_quit_at_logs_on_mtime_change(self) -> None:
        mock_exit = MagicMock()
        checker = quit_at(func=mock_exit)
        with patch("kain.signals.get_mtime") as mock_mtime:
            mock_mtime.return_value = 777777.0
            with patch("kain.signals.logger") as mock_logger:
                checker()
                mock_logger.warning.assert_called()
                args = " ".join(
                    str(c) for c in mock_logger.warning.call_args[0]
                )
                assert "updated at" in args or "stop" in args

    def test_quit_at_sleep_uses_poll_kw_default(self) -> None:
        checker = quit_at(poll=0.05)
        with patch("kain.signals.time.sleep") as mock_sleep:
            checker.sleep(0.1)
            # should sleep at least once with poll=0.05
            assert mock_sleep.call_count >= 1
            assert mock_sleep.call_args[0][0] == pytest.approx(0.05, abs=0.01)

    def test_multiple_quit_at_instances_different_signals(self) -> None:
        mock_exit_a = MagicMock()
        mock_exit_b = MagicMock()
        with patch("kain.signals.bind") as mock_bind:
            checker_a = quit_at(func=mock_exit_a, signal=signal.SIGUSR1)
            checker_b = quit_at(func=mock_exit_b, signal=signal.SIGUSR2)
            # Because quit_at is cached, if args differ we get different objects
            assert checker_a is not checker_b

    def test_on_change_returns_false_on_file_not_found(self) -> None:
        checker = quit_at()
        with patch("kain.signals.get_mtime", side_effect=FileNotFoundError):
            assert checker() is False

    def test_sleep_deadline_zero_returns_true(self) -> None:
        checker = quit_at()
        assert checker.sleep(0) is True


class TestQuitAtParametrized:
    """Parametrized quit_at edge cases."""

    @pytest.mark.parametrize(
        "wait,poll",
        [
            (0.0, 0.0),
            (0.0, 0.1),
            (0.01, 0.001),
            (0.02, 0.005),
        ],
    )
    def test_sleep_various_short_timeouts(
        self,
        wait: float,
        poll: float,
    ) -> None:
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at()
        result = checker.sleep(wait, poll=poll)
        assert result is True

    @pytest.mark.parametrize("errno_val", [0, 1, 137, 255, 999])
    def test_custom_errno_values(self, errno_val: int) -> None:
        import kain.signals as sig_mod

        mock_exit = MagicMock()
        quit_at.cache_clear()  # type: ignore[attr-defined]
        checker = quit_at(
            func=mock_exit,
            signal=signal.SIGUSR1,
            errno=errno_val,
        )
        sig_mod.NeedRestart = True
        try:
            checker()
            mock_exit.assert_called_once_with(errno_val)
        finally:
            sig_mod.NeedRestart = False


class TestOnQuitIntegration:
    """Integration-level tests for on_quit."""

    def test_full_flow_signal_then_teardown_idempotent(self) -> None:
        inst = _fresh_instance()
        cb = MagicMock()
        inst.schedule(cb)
        with pytest.raises(SystemExit):
            inst.signal_handler(signal.SIGTERM, None)
        # teardown already called by signal_handler
        assert inst.already_called is True
        cb.assert_called_once()
        # second teardown does nothing
        inst.teardown()
        cb.assert_called_once()

    def test_full_flow_exception_then_teardown_idempotent(self) -> None:
        inst = _fresh_instance()
        cb = MagicMock()
        inst.schedule(cb)
        inst.exceptions_hooks_proxy(RuntimeError, RuntimeError("boom"), None)
        assert inst.already_called is True
        cb.assert_called_once()
        inst.teardown()
        cb.assert_called_once()

    def test_schedule_after_teardown_ignored(self) -> None:
        inst = _fresh_instance()
        cb = MagicMock()
        inst.teardown()
        inst.schedule(cb)
        # callback was added after teardown, won't run on subsequent teardown
        # because already_called is True
        inst.teardown()
        cb.assert_not_called()

    def test_singleton_instance_persists_across_accesses(self) -> None:
        a = _fresh_instance()
        b = on_quit()
        c = on_quit()
        assert a is b is c

    def test_multiple_callbacks_all_fire(self) -> None:
        calls: list[int] = []
        inst = _fresh_instance()
        for i in range(5):
            inst.schedule(lambda i=i: calls.append(i))
        inst.teardown()
        assert calls == [0, 1, 2, 3, 4]

    def test_exceptions_hooks_proxy_warns_on_original_hook_failure(
        self,
    ) -> None:
        inst = _fresh_instance()
        inst.original_hook = lambda *_: (_ for _ in ()).throw(
            RuntimeError("orig"),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            inst.exceptions_hooks_proxy(ValueError, ValueError("v"), None)
        assert any("orig" in str(m.message) for m in w)

    def test_restore_original_handlers_idempotent(self) -> None:
        inst = _fresh_instance()
        inst.restore_original_handlers()
        # Should not raise when called twice
        inst.restore_original_handlers()
        assert sys.excepthook is inst.original_hook

    def test_threading_handler_delegates_keyboard_interrupt(self) -> None:
        inst = _fresh_instance()
        args = _make_except_hook_args(
            KeyboardInterrupt,
            KeyboardInterrupt("ki"),
            None,
        )
        with patch.object(inst, "_proxy") as proxy:
            inst.threading_handler(args)
            proxy.assert_called_once()

    def test_signal_handler_frame_argument_ignored(self) -> None:
        inst = _fresh_instance()
        with patch.object(inst, "teardown") as mock_teardown:
            with pytest.raises(SystemExit):
                inst.signal_handler(signal.SIGINT, MagicMock())
            mock_teardown.assert_called_once()

    def test_inject_signal_handler_overwrites_existing(self) -> None:
        inst = _fresh_instance()
        with patch("kain.signals.bind") as mock_bind:
            inst.inject_signal_handler()
            assert mock_bind.call_count == 3

    def test_add_hook_returns_none(self) -> None:
        inst = _fresh_instance()
        result = inst.add_hook(lambda *_: None)
        assert result is None
