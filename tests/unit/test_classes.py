from __future__ import annotations

import threading
from typing import TYPE_CHECKING, get_type_hints

import pytest

from kain.classes import Missing, Nothing, Singleton

if TYPE_CHECKING:
    from faker import Faker


@pytest.mark.unit
def test_missing_bool_always_returns_false() -> None:
    """
    Given: a Missing sentinel instance
    When: converting to bool
    Then: returns False
    """
    # --- Arrange ---
    missing = Missing()

    # --- Act ---
    result = bool(missing)

    # --- Assert ---
    assert result is False


@pytest.mark.unit
@pytest.mark.parametrize(
    "target",
    (
        pytest.param(None, id="none"),
        pytest.param(False, id="false"),
        pytest.param(0, id="zero"),
        pytest.param("", id="empty-string"),
        pytest.param([], id="empty-list"),
        pytest.param(object(), id="object"),
    ),
)
def test_missing_eq_with_any_object_returns_false(
    target: object,
    fake: Faker,
) -> None:
    """
    Given: a Missing sentinel instance
    When: comparing it to any object
    Then: returns False
    """
    # --- Arrange ---
    missing = Missing()

    # --- Act ---
    result = missing == target

    # --- Assert ---
    assert result is False


@pytest.mark.unit
def test_missing_eq_with_self_returns_false() -> None:
    """
    Given: a Missing sentinel instance
    When: comparing it to another Missing instance
    Then: returns False
    """
    # --- Arrange ---
    first = Missing()
    second = Missing()

    # --- Act ---
    result = first == second

    # --- Assert ---
    assert result is False


@pytest.mark.unit
def test_missing_hash_uses_id() -> None:
    """
    Given: a Missing sentinel instance
    When: hashing it
    Then: hash equals id
    """
    # --- Arrange ---
    missing = Missing()

    # --- Act ---
    result = hash(missing)

    # --- Assert ---
    assert result == id(missing)


@pytest.mark.unit
def test_missing_is_hashable_as_dict_key(fake: Faker) -> None:
    """
    Given: a Missing sentinel instance
    When: using it as a dict key
    Then: no TypeError is raised
    """
    # --- Arrange ---
    missing = Missing()
    expected = fake.pyint()

    # --- Act ---
    mapping = {missing: expected}

    # --- Assert ---
    assert mapping[missing] == expected


@pytest.mark.unit
def test_missing_repr_contains_class_name() -> None:
    """
    Given: a Missing sentinel instance
    When: calling repr
    Then: contains "Missing"
    """
    # --- Arrange ---
    missing = Missing()

    # --- Act ---
    result = repr(missing)

    # --- Assert ---
    assert "Missing" in result


@pytest.mark.unit
def test_nothing_is_missing_instance() -> None:
    """
    Given: the Nothing sentinel
    When: checking type and bool
    Then: isinstance of Missing and bool is False
    """
    # --- Arrange ---
    nothing = Nothing

    # --- Act / Assert ---
    assert isinstance(nothing, Missing)
    assert bool(nothing) is False


@pytest.mark.unit
def test_singleton_first_call_creates_instance(fake: Faker) -> None:
    """
    Given: a Singleton-metaclass class
    When: calling it for the first time
    Then: a new instance is created and cls.instance is set
    """

    # --- Arrange ---
    class Sample(metaclass=Singleton):
        def __init__(self, value: int = 0) -> None:
            self.value = value

    # --- Act ---
    instance = Sample(fake.pyint())

    # --- Assert ---
    assert instance is not Nothing
    assert Sample.instance is instance


@pytest.mark.unit
def test_singleton_second_call_returns_same_instance(fake: Faker) -> None:
    """
    Given: a Singleton-metaclass class already instantiated
    When: calling it a second time with different args
    Then: the same instance is returned
    """

    # --- Arrange ---
    class Sample(metaclass=Singleton):
        def __init__(self, value: int = 0) -> None:
            self.value = value

    first = Sample(fake.pyint())

    # --- Act ---
    second = Sample(fake.pyint(min_value=1000, max_value=9999))

    # --- Assert ---
    assert first is second


@pytest.mark.unit
def test_singleton_ignores_new_args_on_subsequent_calls(
    fake: Faker,
) -> None:
    """
    Given: a Singleton-metaclass class instantiated with arg X
    When: calling it again with arg Y
    Then: the cached instance retains the original value
    """

    # --- Arrange ---
    class Sample(metaclass=Singleton):
        def __init__(self, value: int = 0) -> None:
            self.value = value

    original_value = fake.pyint()
    first = Sample(original_value)

    # --- Act ---
    second = Sample(fake.pyint(min_value=1000, max_value=9999))

    # --- Assert ---
    assert second.value == original_value
    assert first is second


@pytest.mark.unit
def test_singleton_thread_safety_race_condition(fake: Faker) -> None:
    """
    Given: a Singleton-metaclass class
    When: instantiating from multiple threads concurrently
    Then: all threads receive the exact same instance
    """

    # --- Arrange ---
    class Sample(metaclass=Singleton):
        def __init__(self, value: int = 0) -> None:
            self.value = value

    results: list[object] = []
    lock = threading.Lock()

    def _create() -> None:
        instance = Sample(fake.pyint())
        with lock:
            results.append(instance)

    threads = [threading.Thread(target=_create) for _ in range(20)]

    # --- Act ---
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # --- Assert ---
    assert len({id(r) for r in results}) == 1


@pytest.mark.unit
def test_singleton_instance_attribute_is_set(fake: Faker) -> None:
    """
    Given: a Singleton-metaclass class
    When: after first instantiation
    Then: cls.instance equals the created object
    """

    # --- Arrange ---
    class Sample(metaclass=Singleton):
        pass

    # --- Act ---
    instance = Sample()

    # --- Assert ---
    assert Sample.instance is instance


# ------------------------------------------------------------------
# Annotation inference
# ------------------------------------------------------------------


class TestAnnotationInference:
    """GIVEN Missing and Singleton methods
    WHEN inspecting type hints
    THEN signatures match expected types.
    """

    def test_missing_hash_returns_int(self) -> None:
        """GIVEN Missing.__hash__
        WHEN getting type hints
        THEN return type is int.
        """
        # --- Act ---
        hints = get_type_hints(Missing.__hash__)

        # --- Assert ---
        assert hints["return"] is int

    def test_missing_eq_params_and_return(self) -> None:
        """GIVEN Missing.__eq__
        WHEN getting type hints
        THEN param is object and return is bool.
        """
        # --- Act ---
        hints = get_type_hints(Missing.__eq__)

        # --- Assert ---
        assert hints["_"] is object
        assert hints["return"] is bool

    def test_singleton_call_returns_object(self) -> None:
        """GIVEN Singleton.__call__
        WHEN getting type hints
        THEN return type is object.
        """
        # --- Act ---
        hints = get_type_hints(Singleton.__call__)

        # --- Assert ---
        assert hints["return"] is object


# ------------------------------------------------------------------
# Inheritance contract
# ------------------------------------------------------------------


class TestInheritanceContract:
    """GIVEN Singleton as a metaclass
    WHEN a class uses it
    THEN inherited behavior (instance caching) works on subclasses.
    """

    def test_subclass_inherits_singleton_caching(self, fake: Faker) -> None:
        """GIVEN a base class with Singleton
        WHEN a subclass is instantiated
        THEN the subclass also caches its instance.
        """

        # --- Arrange ---
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        # --- Act ---
        first = Child()
        second = Child()

        # --- Assert ---
        assert first is second
        assert Child.instance is first

    def test_subclass_instance_is_independent_of_base(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a base Singleton class and its subclass
        WHEN both are instantiated
        THEN each has its own cached instance.
        """

        # --- Arrange ---
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        # --- Act ---
        base_instance = Base()
        child_instance = Child()

        # --- Assert ---
        assert base_instance is not child_instance
        assert Base.instance is base_instance
        assert Child.instance is child_instance


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Paranoid edge-case coverage for Missing and Singleton."""

    def test_missing_instances_have_unique_hashes(self) -> None:
        """GIVEN two Missing instances
        WHEN hashing both
        THEN hashes differ because they are distinct objects.
        """
        # --- Arrange ---
        first = Missing()
        second = Missing()

        # --- Act ---
        first_hash = hash(first)
        second_hash = hash(second)

        # --- Assert ---
        assert first_hash != second_hash
        assert first_hash == id(first)
        assert second_hash == id(second)

    def test_missing_in_set_deduplication(self) -> None:
        """GIVEN Missing instances in a set
        WHEN adding multiple instances
        THEN each is kept because hashes differ.
        """
        # --- Arrange ---
        expected_count = 3
        items = [Missing() for _ in range(expected_count)]

        # --- Act ---
        container = set(items)

        # --- Assert ---
        assert len(container) == expected_count

    def test_singleton_init_exception_leaves_instance_as_nothing(self) -> None:
        """GIVEN a Singleton class whose __init__ raises
        WHEN instantiation is attempted
        THEN cls.instance remains Nothing.
        """

        # --- Arrange ---
        class Broken(metaclass=Singleton):
            def __init__(self) -> None:
                msg = "always fails"
                raise RuntimeError(msg)

        # --- Act / Assert ---
        with pytest.raises(RuntimeError):
            Broken()

        assert Broken.instance is Nothing

    def test_singleton_recovers_after_failed_init(self, fake: Faker) -> None:
        """GIVEN a Singleton that failed once
        WHEN instantiated again after fixing
        THEN it succeeds and caches the new instance.
        """

        # --- Arrange ---
        call_count = 0

        class Flaky(metaclass=Singleton):
            def __init__(self) -> None:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    msg = "first call fails"
                    raise RuntimeError(msg)

        # --- Act ---
        with pytest.raises(RuntimeError):
            Flaky()

        instance = Flaky()

        # --- Assert ---
        assert instance is not Nothing
        assert Flaky.instance is instance
        assert call_count == 2

    def test_nothing_repr_contains_class_name(self) -> None:
        """GIVEN the Nothing sentinel
        WHEN calling repr
        THEN it contains 'Missing'.
        """
        # --- Act ---
        result = repr(Nothing)

        # --- Assert ---
        assert "Missing" in result
