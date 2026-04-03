"""Tests for kain.classes module."""

from __future__ import annotations

from kain.classes import Missing, Nothing, Singleton


class TestMissing:
    """Tests for Missing sentinel class."""

    def test_bool_is_false(self) -> None:
        """Missing instances should be falsy."""
        missing = Missing()
        assert bool(missing) is False
        assert not missing

    def test_eq_always_false(self) -> None:
        """Missing instances should never compare equal to anything."""
        missing = Missing()
        assert missing != missing
        assert missing != Nothing
        assert missing != None  # noqa: E711
        assert missing != 42
        assert missing != ""
        assert missing != []

    def test_hash_returns_id(self) -> None:
        """__hash__ should return the object's id."""
        missing = Missing()
        assert hash(missing) == id(missing)

    def test_repr_contains_name_and_addr(self) -> None:
        """__repr__ should include class name and memory address."""
        missing = Missing()
        rep = repr(missing)
        assert "Missing" in rep
        assert f"#{id(missing):x}" in rep


class TestNothing:
    """Tests for the Nothing singleton sentinel."""

    def test_is_missing_instance(self) -> None:
        """Nothing should be an instance of Missing."""
        assert isinstance(Nothing, Missing)

    def test_is_falsy(self) -> None:
        """Nothing should be falsy."""
        assert bool(Nothing) is False

    def test_singleton_identity(self) -> None:
        """Nothing should always be the same object."""
        from kain.classes import Nothing as Nothing2

        assert Nothing is Nothing2


class TestSingleton:
    """Tests for the Singleton metaclass."""

    def test_returns_same_instance(self) -> None:
        """A class using Singleton should return the same instance."""

        class Single(metaclass=Singleton):
            pass

        a = Single()
        b = Single()
        assert a is b

    def test_instance_created_once(self) -> None:
        """__init__ should only be called once for the singleton."""
        calls: list[tuple[object, ...]] = []

        class Single(metaclass=Singleton):
            def __init__(self, value: object) -> None:
                calls.append((value,))

        s1 = Single(1)
        s2 = Single(2)
        assert s1 is s2
        assert len(calls) == 1
        assert calls[0] == (1,)

    def test_instance_attribute_reset(self) -> None:
        """Resetting the metaclass instance attribute creates a new singleton."""

        class Single(metaclass=Singleton):
            pass

        a = Single()
        Single.instance = Nothing
        b = Single()
        assert a is not b
        c = Single()
        assert b is c
