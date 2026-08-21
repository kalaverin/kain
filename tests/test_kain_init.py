"""Tests for the public API exported by ``kain``."""

from __future__ import annotations

import inspect
import types

import kain


class TestKainAll:
    """Tests for ``kain.__all__`` consistency."""

    def test_all_is_tuple(self) -> None:
        """``__all__`` must be a tuple."""
        assert isinstance(kain.__all__, tuple)

    def test_all_length(self) -> None:
        """``__all__`` should contain exactly 9 public names."""
        assert len(kain.__all__) == 9

    def test_all_has_no_duplicates(self) -> None:
        """No name should appear more than once in ``__all__``."""
        assert len(kain.__all__) == len(set(kain.__all__))

    def test_all_matches_expected(self) -> None:
        """``__all__`` should match the known public API set."""
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
        assert set(kain.__all__) == expected

    def test_all_names_are_accessible(self) -> None:
        """Every name listed in ``__all__`` must exist as a module
        attribute."""
        for name in kain.__all__:
            assert hasattr(kain, name)


class TestKainExportTypes:
    """Tests verifying the type of each public export."""

    def test_monkey_is_class(self) -> None:
        """``Monkey`` is a namespace class."""
        assert inspect.isclass(kain.Monkey)

    def test_add_path_is_function(self) -> None:
        """``add_path`` is a plain function."""
        assert isinstance(kain.add_path, types.FunctionType)

    def test_optional_is_function(self) -> None:
        """``optional`` is a plain function."""
        assert isinstance(kain.optional, types.FunctionType)

    def test_required_is_function(self) -> None:
        """``required`` is a plain function."""
        assert isinstance(kain.required, types.FunctionType)

    def test_to_ascii_is_function(self) -> None:
        """``to_ascii`` is a plain function."""
        assert isinstance(kain.to_ascii, types.FunctionType)

    def test_to_bytes_is_function(self) -> None:
        """``to_bytes`` is a plain function."""
        assert isinstance(kain.to_bytes, types.FunctionType)

    def test_unique_is_function(self) -> None:
        """``unique`` is a plain function."""
        assert isinstance(kain.unique, types.FunctionType)


class TestKainExportBehavior:
    """Lightweight behavioral sanity checks for exported objects."""

    def test_monkey_has_patch_bind_wrap(self) -> None:
        """``Monkey`` must expose its three documented classmethods."""
        assert callable(kain.Monkey.replace)
        assert callable(kain.Monkey.bind)
        assert callable(kain.Monkey.wrap)
