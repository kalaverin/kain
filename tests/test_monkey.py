"""Tests for kain.monkey module."""

from __future__ import annotations

import logging
import types
from unittest.mock import MagicMock, patch

import pytest

from kain.monkey import Monkey


@pytest.fixture(autouse=True)
def _isolate_monkey_state() -> None:
    """Save and restore Monkey.mapping around each test."""
    original_mapping = dict(Monkey.mapping)
    Monkey.mapping.clear()
    yield
    Monkey.mapping.clear()
    Monkey.mapping.update(original_mapping)


class TestMonkeyExpect:
    """Tests for Monkey.expect decorator."""

    def test_suppresses_expected_exceptions(self) -> None:
        """Should suppress the specified exceptions."""

        class Kls:
            @Monkey.expect(ValueError)
            def boom(cls) -> None:
                raise ValueError("boom")

        # Should not raise
        Kls.boom()

    def test_does_not_suppress_other_exceptions(self) -> None:
        """Should not suppress unexpected exceptions."""

        class Kls:
            @Monkey.expect(ValueError)
            def boom(cls) -> None:
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            Kls.boom()

    def test_returns_value_when_no_exception(self) -> None:
        """Should return the function result normally."""

        class Kls:
            @Monkey.expect(ValueError)
            def ok(cls) -> int:
                return 42

        assert Kls.ok() == 42


class TestMonkeyPatch:
    """Tests for Monkey.patch."""

    def test_patch_with_tuple(self) -> None:
        """Should patch an attribute given a (node, name) tuple."""
        node = types.SimpleNamespace(key="old")
        replacement = "new"

        result = Monkey.patch((node, "key"), replacement)

        assert result is replacement
        assert node.key is replacement
        assert Monkey.mapping[replacement] == "old"

        # Restore
        node.key = Monkey.mapping.pop(replacement)

    def test_patch_with_module_object(self) -> None:
        """Should patch a module attribute when passed a module."""
        mod = types.ModuleType("test_mod")
        original = MagicMock()
        original.__name__ = "original_attr"
        replacement = MagicMock()
        replacement.__name__ = "original_attr"
        setattr(mod, "original_attr", original)

        def fake_required(path: object, *args: object) -> object:
            if args:
                return original
            return mod

        with patch("kain.monkey.Is.module", return_value=True):
            with patch("kain.monkey.required", side_effect=fake_required):
                result = Monkey.patch(mod, replacement)

        assert result is replacement
        assert getattr(mod, "original_attr") is replacement
        assert Monkey.mapping[replacement] is original

        # Restore
        setattr(mod, "original_attr", Monkey.mapping.pop(replacement))

    def test_patch_with_string_path(self) -> None:
        """Should patch an attribute specified by dotted string path."""
        mod = types.ModuleType("test_patch_mod")
        original = lambda: "old"  # noqa: E731
        setattr(mod, "func", original)
        replacement = lambda: "new"  # noqa: E731

        def fake_required(path: object, *args: object) -> object:
            if args:
                return original
            return mod

        with patch("kain.monkey.required", side_effect=fake_required):
            result = Monkey.patch("test_patch_mod.func", replacement)

        assert result is replacement
        assert getattr(mod, "func") is replacement
        assert Monkey.mapping[replacement] is original

        # Restore
        setattr(mod, "func", Monkey.mapping.pop(replacement))

    def test_patch_returns_same_if_already_set(self) -> None:
        """Should return new immediately if it is already the attribute."""
        node = types.SimpleNamespace(key="value")
        result = Monkey.patch((node, "key"), "value")
        assert result == "value"
        assert "value" not in Monkey.mapping

    def test_patch_raises_when_old_is_new(self) -> None:
        """Should raise RuntimeError when old and new are the same."""
        node = types.SimpleNamespace()

        def fake_required(path: object, *args: object) -> object:
            return node

        with patch("kain.monkey.required", side_effect=fake_required):
            with pytest.raises(RuntimeError):
                Monkey.patch((node, "func"), node)

    def test_patch_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log a debug message on successful patch."""
        caplog.set_level(logging.DEBUG, logger="kain.monkey")
        node = types.SimpleNamespace(key="old")
        replacement = "new"

        Monkey.patch((node, "key"), replacement)
        assert "->" in caplog.text

        node.key = Monkey.mapping.pop(replacement)


class TestMonkeyBind:
    """Tests for Monkey.bind."""

    def test_bind_without_decorator(self) -> None:
        """Should bind a plain function to a node."""
        node = types.SimpleNamespace()

        @Monkey.bind(node)
        def helper() -> int:
            return 42

        assert node.helper() == 42  # type: ignore[attr-defined]

    def test_bind_with_custom_name(self) -> None:
        """Should use the provided name when binding."""
        node = types.SimpleNamespace()

        @Monkey.bind(node, name="custom")
        def helper() -> int:
            return 42

        assert node.custom() == 42  # type: ignore[attr-defined]

    def test_bind_with_classmethod_decorator(self) -> None:
        """Should pass the node as the first arg when decorator is classmethod."""
        node = types.SimpleNamespace()
        received: list[object] = []

        @Monkey.bind(node, decorator=classmethod)
        def helper(target: object) -> object:
            received.append(target)
            return target

        result = node.helper()  # type: ignore[attr-defined]
        assert result is node
        assert received == [node]


class TestMonkeyWrap:
    """Tests for Monkey.wrap."""

    def test_wrap_without_decorator(self) -> None:
        """Should wrap an existing method and pass it as the first argument."""
        node = types.SimpleNamespace()

        def original(x: int) -> int:
            return x * 2

        node.mul = original
        calls: list[tuple[object, ...]] = []

        @Monkey.wrap(node, "mul")
        def wrapper(wrapped: object, x: int) -> int:
            calls.append(("before", x))
            result = wrapped(x)  # type: ignore[operator]
            calls.append(("after", result))
            return result

        assert node.mul(5) == 10  # type: ignore[operator]
        assert calls == [("before", 5), ("after", 10)]

        # Restore
        patched = node.mul
        if patched in Monkey.mapping:
            node.mul = Monkey.mapping.pop(patched)
        else:
            node.mul = original

    def test_wrap_with_decorator(self) -> None:
        """Should apply an optional decorator to the wrapper."""
        node = types.SimpleNamespace()

        def original() -> str:
            return "original"

        node.func = original
        calls: list[object] = []

        def my_decorator(fn: object) -> object:
            calls.append("decorated")
            return fn

        @Monkey.wrap(node, "func", decorator=my_decorator)
        def wrapper(wrapped: object) -> str:
            return "wrapped:" + wrapped()  # type: ignore[operator]

        assert node.func() == "wrapped:original"  # type: ignore[operator]
        assert "decorated" in calls

        # Restore
        patched = node.func
        if patched in Monkey.mapping:
            node.func = Monkey.mapping.pop(patched)
        else:
            node.func = original

    def test_wrap_on_class(self) -> None:
        """Should wrap a class method."""

        class Node:
            def method(self) -> str:
                return "original"

        original_method = Node.method

        @Monkey.wrap(Node, "method")
        def wrapper(wrapped: object, self: object) -> str:
            return "wrapped:" + wrapped(self)  # type: ignore[operator]

        obj = Node()
        assert obj.method() == "wrapped:original"

        # Restore class method
        patched = Node.method
        if patched in Monkey.mapping:
            Node.method = Monkey.mapping.pop(patched)
        else:
            Node.method = original_method  # type: ignore[assignment]

    def test_wrap_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log an info message when wrapping."""
        caplog.set_level(logging.INFO, logger="kain.monkey")
        node = types.SimpleNamespace()
        node.func = lambda: None

        @Monkey.wrap(node, "func")
        def wrapper(wrapped: object) -> None:
            return None

        assert "func" in caplog.text

        # Restore
        patched = node.func
        if patched in Monkey.mapping:
            node.func = Monkey.mapping.pop(patched)
        else:
            node.func = lambda: None
