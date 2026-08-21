"""Tests for the public API exported by ``zx.core``."""

from __future__ import annotations

import inspect
import types

import zx
import zx.core


class TestZxNamespace:
    """Tests for the ``zx`` implicit namespace package."""

    def test_has_no_regular_package_origin(self) -> None:
        assert zx.__spec__.origin is None

    def test_exposes_core_after_import(self) -> None:
        assert zx.core is zx.core


class TestZxCoreAll:
    """Tests for ``zx.core.__all__`` consistency."""

    def test_all_is_tuple(self) -> None:
        assert isinstance(zx.core.__all__, tuple)

    def test_all_matches_expected(self) -> None:
        expected = {
            "Is",
            "Monkey",
            "Who",
            "add_path",
            "optional",
            "required",
            "to_ascii",
            "to_bytes",
            "unique",
        }
        assert set(zx.core.__all__) == expected

    def test_all_names_are_accessible(self) -> None:
        for name in zx.core.__all__:
            assert hasattr(zx.core, name)


class TestZxCoreExportTypes:
    """Tests for public export types."""

    def test_monkey_is_class(self) -> None:
        assert inspect.isclass(zx.core.Monkey)

    def test_add_path_is_function(self) -> None:
        assert isinstance(zx.core.add_path, types.FunctionType)

    def test_optional_is_function(self) -> None:
        assert isinstance(zx.core.optional, types.FunctionType)

    def test_required_is_function(self) -> None:
        assert isinstance(zx.core.required, types.FunctionType)

    def test_to_ascii_is_function(self) -> None:
        assert isinstance(zx.core.to_ascii, types.FunctionType)

    def test_to_bytes_is_function(self) -> None:
        assert isinstance(zx.core.to_bytes, types.FunctionType)

    def test_unique_is_function(self) -> None:
        assert isinstance(zx.core.unique, types.FunctionType)


class TestZxCoreExportBehavior:
    """Tests for lightweight public export behavior."""

    def test_monkey_has_replace_bind_wrap(self) -> None:
        assert callable(zx.core.Monkey.replace)
        assert callable(zx.core.Monkey.bind)
        assert callable(zx.core.Monkey.wrap)
