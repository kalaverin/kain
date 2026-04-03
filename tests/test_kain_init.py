"""Tests for the public API exported by ``kain``."""

from __future__ import annotations

import inspect
import types

import kain
from kain.classes import Singleton


class TestKainAll:
    """Tests for ``kain.__all__`` consistency."""

    def test_all_is_tuple(self) -> None:
        """``__all__`` must be a tuple."""
        assert isinstance(kain.__all__, tuple)

    def test_all_length(self) -> None:
        """``__all__`` should contain exactly 18 public names."""
        assert len(kain.__all__) == 18

    def test_all_has_no_duplicates(self) -> None:
        """No name should appear more than once in ``__all__``."""
        assert len(kain.__all__) == len(set(kain.__all__))

    def test_all_matches_expected(self) -> None:
        """``__all__`` should match the known public API set."""
        expected = {
            "Is",
            "Missing",
            "Monkey",
            "Nothing",
            "Who",
            "add_path",
            "on_quit",
            "optional",
            "quit_at",
            "required",
            "sort",
            "to_ascii",
            "to_bytes",
            "unique",
            "cache",
            "class_property",
            "mixed_property",
            "pin",
        }
        assert set(kain.__all__) == expected

    def test_all_names_are_accessible(self) -> None:
        """Every name listed in ``__all__`` must exist as a module attribute."""
        for name in kain.__all__:
            assert hasattr(kain, name)


class TestKainExportTypes:
    """Tests verifying the type of each public export."""

    def test_is_is_class(self) -> None:
        """``Is`` is a dataclass-class namespace."""
        assert inspect.isclass(kain.Is)

    def test_missing_is_class(self) -> None:
        """``Missing`` is a sentinel class."""
        assert inspect.isclass(kain.Missing)

    def test_monkey_is_class(self) -> None:
        """``Monkey`` is a namespace class."""
        assert inspect.isclass(kain.Monkey)

    def test_nothing_is_missing_instance(self) -> None:
        """``Nothing`` is the global singleton instance of ``Missing``."""
        assert isinstance(kain.Nothing, kain.Missing)
        assert kain.Nothing is not kain.Missing()

    def test_who_is_class(self) -> None:
        """``Who`` is a dataclass-class namespace."""
        assert inspect.isclass(kain.Who)

    def test_add_path_is_function(self) -> None:
        """``add_path`` is a plain function."""
        assert isinstance(kain.add_path, types.FunctionType)

    def test_on_quit_is_singleton_class(self) -> None:
        """``on_quit`` is a class using the ``Singleton`` metaclass."""
        assert inspect.isclass(kain.on_quit)
        assert isinstance(kain.on_quit, Singleton)

    def test_optional_is_function(self) -> None:
        """``optional`` is a plain function."""
        assert isinstance(kain.optional, types.FunctionType)

    def test_quit_at_is_callable(self) -> None:
        """``quit_at`` is a callable (cached function)."""
        assert callable(kain.quit_at)

    def test_required_is_function(self) -> None:
        """``required`` is a plain function."""
        assert isinstance(kain.required, types.FunctionType)

    def test_sort_is_callable(self) -> None:
        """``sort`` resolves to a callable (either ``natsorted`` or ``sorted``)."""
        assert callable(kain.sort)

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

    def test_nothing_is_falsy(self) -> None:
        """The sentinel ``Nothing`` must be falsy."""
        assert bool(kain.Nothing) is False

    def test_missing_instance_never_equal(self) -> None:
        """Two distinct ``Missing`` instances must not compare equal."""
        assert kain.Missing() != kain.Missing()

    def test_monkey_has_expect_patch_bind_wrap(self) -> None:
        """``Monkey`` must expose its four documented classmethods."""
        assert callable(kain.Monkey.expect)
        assert callable(kain.Monkey.patch)
        assert callable(kain.Monkey.bind)
        assert callable(kain.Monkey.wrap)
