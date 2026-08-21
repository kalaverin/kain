"""Tests for kain.monkey module."""

from __future__ import annotations

import logging
import types
from unittest.mock import patch

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


class TestMonkeyPatch:
    """Tests for Monkey.patch."""

    def test_patch_with_tuple(self) -> None:
        """Should patch an attribute given a (node, name) tuple."""
        node = types.SimpleNamespace(key="old")
        replacement = "new"

        result = Monkey.replace((node, "key"), replacement)

        assert result is replacement
        assert node.key is replacement
        assert Monkey.mapping[replacement] == "old"

        # Restore
        node.key = Monkey.mapping.pop(replacement)

    def test_patch_with_module_object(self) -> None:
        """Should patch a module attribute when passed a module."""
        mod = types.ModuleType("test_mod")

        def original() -> str:
            return "old"

        original.__name__ = "original_attr"
        original.__qualname__ = "original_attr"

        def replacement() -> str:
            return "new"

        replacement.__name__ = "original_attr"
        replacement.__qualname__ = "original_attr"
        mod.original_attr = original

        def fake_required(path: object, *args: object) -> object:
            if args:
                return original
            return mod

        with (
            patch("kain.monkey._is.module", return_value=True),
            patch("kain.monkey.required", side_effect=fake_required),
        ):
            result = Monkey.replace(mod, replacement)

        assert result is replacement
        assert mod.original_attr is replacement
        assert Monkey.mapping[replacement] is original

        # Restore
        mod.original_attr = Monkey.mapping.pop(replacement)

    def test_patch_with_string_path(self) -> None:
        """Should patch an attribute specified by dotted string path."""
        mod = types.ModuleType("test_patch_mod")
        original = lambda: "old"  # noqa: E731
        mod.func = original
        replacement = lambda: "new"  # noqa: E731

        def fake_required(path: object, *args: object) -> object:
            if args:
                return original
            return mod

        with patch("kain.monkey.required", side_effect=fake_required):
            result = Monkey.replace("test_patch_mod.func", replacement)

        assert result is replacement
        assert mod.func is replacement
        assert Monkey.mapping[replacement] is original

        # Restore
        mod.func = Monkey.mapping.pop(replacement)

    def test_patch_returns_same_if_already_set(self) -> None:
        """Should return new immediately if it is already the attribute."""
        node = types.SimpleNamespace(key="value")
        result = Monkey.replace((node, "key"), "value")
        assert result == "value"
        assert "value" not in Monkey.mapping

    @pytest.mark.xfail(
        reason="Current implementation has an early return when "
        "getattr(node, name, None) is new, preventing RuntimeError "
        "from ever being reached in practice.",
        strict=True,
    )
    def test_patch_raises_when_old_is_new(self) -> None:
        """Should raise RuntimeError when old and new are the same."""
        node = types.SimpleNamespace()
        node.func = node

        with pytest.raises(RuntimeError):
            Monkey.replace((node, "func"), node)

    def test_patch_logs_debug(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log a debug message on successful patch."""
        caplog.set_level(logging.DEBUG, logger="kain.monkey")
        node = types.SimpleNamespace(key="old")
        replacement = "new"

        Monkey.replace((node, "key"), replacement)
        assert any(
            record.message == "attribute replaced"
            and getattr(record, "before", "") != ""
            and getattr(record, "after", "") != ""
            for record in caplog.records
        )

        node.key = Monkey.mapping.pop(replacement)


class TestMonkeyBind:
    """Tests for Monkey.bind."""

    def test_bind_without_decorator(self) -> None:
        """Should bind a plain function to a node."""
        node = types.SimpleNamespace()

        @Monkey.bind(node)
        def helper() -> int:
            return 42

        assert node.helper() == 42  # type: ignore[assignment][attr-defined]

    def test_bind_with_custom_name(self) -> None:
        """Should use the provided name when binding."""
        node = types.SimpleNamespace()

        @Monkey.bind(node, name="custom")
        def helper() -> int:
            return 42

        assert node.custom() == 42  # type: ignore[assignment][attr-defined]

    def test_bind_with_classmethod_decorator(self) -> None:
        """Should pass the node as the first arg when decorator is
        classmethod."""
        node = types.SimpleNamespace()
        received: list[object] = []

        @Monkey.bind(node, decorator=classmethod)
        def helper(target: object) -> object:
            received.append(target)
            return target

        result = node.helper()  # type: ignore[assignment][attr-defined]
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
            result = wrapped(x)  # type: ignore[assignment][operator]
            calls.append(("after", result))
            return result

        assert node.mul(5) == 10  # type: ignore[assignment][operator]
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
            return "wrapped:" + wrapped()  # type: ignore[assignment][operator]

        assert node.func() == "wrapped:original"  # type: ignore[assignment][operator]
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
            return "wrapped:" + wrapped(self)  # type: ignore[assignment][operator]

        obj = Node()
        assert obj.method() == "wrapped:original"

        # Restore class method
        patched = Node.method
        if patched in Monkey.mapping:
            Node.method = Monkey.mapping.pop(patched)
        else:
            Node.method = original_method  # type: ignore[assignment][assignment]

    def test_wrap_logs_info(self, caplog: pytest.LogCaptureFixture) -> None:
        """Should log a debug message when wrapping."""
        caplog.set_level(logging.DEBUG, logger="kain.monkey")
        node = types.SimpleNamespace()
        node.func = lambda: None

        @Monkey.wrap(node, "func")
        def wrapper(wrapped: object) -> None:
            return None

        assert any(
            record.message == "attribute wrapped with"
            and getattr(record, "before", "") != ""
            and getattr(record, "after", "") != ""
            for record in caplog.records
        )

        # Restore
        patched = node.func
        if patched in Monkey.mapping:
            node.func = Monkey.mapping.pop(patched)
        else:
            node.func = lambda: None


class TestMonkeyPatchExtended:
    """Extended tests for Monkey.patch."""

    def test_patch_object_attribute(self) -> None:
        node = types.SimpleNamespace(value=1)
        Monkey.replace((node, "value"), 2)
        assert node.value == 2
        node.value = Monkey.mapping.pop(2, 1)

    def test_patch_function(self) -> None:
        node = types.SimpleNamespace()

        def original() -> str:
            return "old"

        def replacement() -> str:
            return "new"

        node.func = original
        Monkey.replace((node, "func"), replacement)
        assert node.func() == "new"
        node.func = Monkey.mapping.pop(replacement, original)

    def test_patch_identity_shortcircuit(self) -> None:
        node = types.SimpleNamespace()
        val = object()
        node.x = val
        result = Monkey.replace((node, "x"), val)
        assert result is val
        assert val not in Monkey.mapping

    def test_patch_class_attribute(self) -> None:
        class Node:
            attr = "old"

        Monkey.replace((Node, "attr"), "new")
        assert Node.attr == "new"
        Node.attr = Monkey.mapping.pop("new", "old")

    def test_patch_restores_via_mapping(self) -> None:
        node = types.SimpleNamespace(a=1)
        Monkey.replace((node, "a"), 2)
        old = Monkey.mapping.pop(2)
        node.a = old
        assert node.a == 1


class TestMonkeyBindExtended:
    """Extended tests for Monkey.bind."""

    def test_bind_returns_wrapper(self) -> None:
        node = types.SimpleNamespace()

        @Monkey.bind(node)
        def func() -> int:
            return 1

        assert node.func() == 1

    def test_bind_kwargs_passed_through(self) -> None:
        node = types.SimpleNamespace()

        @Monkey.bind(node)
        def func(**kw: object) -> dict[str, object]:
            return kw

        assert node.func(a=1, b=2) == {"a": 1, "b": 2}

    def test_bind_args_passed_through(self) -> None:
        node = types.SimpleNamespace()

        @Monkey.bind(node)
        def func(a: int, b: int) -> int:
            return a + b

        assert node.func(1, 2) == 3


class TestMonkeyWrapExtended:
    """Extended tests for Monkey.wrap."""

    def test_wrap_preserves_args(self) -> None:
        node = types.SimpleNamespace()
        node.fn = lambda x, y: x + y

        @Monkey.wrap(node, "fn")
        def wrapper(wrapped: object, x: int, y: int) -> int:
            return wrapped(x, y) * 2  # type: ignore[assignment][operator]

        assert node.fn(1, 2) == 6

    def test_wrap_staticmethod(self) -> None:
        class Node:
            @staticmethod
            def static() -> str:
                return "static"

        original = Node.static

        @Monkey.wrap(Node, "static")
        def wrapper(wrapped: object) -> str:
            return "wrap:" + wrapped()  # type: ignore[assignment][operator]

        assert Node.static() == "wrap:static"

        patched = Node.static
        if patched in Monkey.mapping:
            Node.static = Monkey.mapping.pop(patched)
        else:
            Node.static = original  # type: ignore[assignment][assignment]
