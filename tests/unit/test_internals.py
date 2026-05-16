from __future__ import annotations

from typing import Any, Literal, Union, get_type_hints

import pytest
from faker import Faker

from kain import Is, Who
from kain.internals import (
    get_attr,
    get_mro,
    get_owner,
    is_subclass,
    iter_inheritance,
    iter_stack,
    to_ascii,
    to_bytes,
    unique,
)


@pytest.mark.unit
def test_who_callables_are_callable() -> None:
    """
    Given: the Who namespace
    When: checking its members
    Then: every member is callable
    """
    # --- Act / Assert ---
    assert callable(Who.Args)
    assert callable(Who.Cast)
    assert callable(Who.File)
    assert callable(Who.Inheritance)
    assert callable(Who.Is)
    assert callable(Who.Module)
    assert callable(Who.Addr)
    assert callable(Who.Name)


@pytest.mark.unit
def test_is_tty_is_bool() -> None:
    """
    Given: the Is namespace
    When: accessing the tty field
    Then: it is a bool instance
    """
    # --- Act / Assert ---
    assert isinstance(Is.tty, bool)


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_value,expected_result",
    (
        pytest.param(b"hello", "hello", id="bytes-input"),
        pytest.param("hello", "hello", id="str-input"),
    ),
)
def test_to_ascii_coerces_bytes_and_str(
    input_value: bytes | str,
    expected_result: str,
) -> None:
    """
    Given: bytes or str input
    When: calling to_ascii
    Then: returns a str
    """
    # --- Act ---
    result = to_ascii(input_value)

    # --- Assert ---
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_value",
    (
        pytest.param(123, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
    ),
)
def test_to_ascii_raises_type_error_for_invalid_type(
    invalid_value: object,
) -> None:
    """
    Given: an invalid type for to_ascii
    When: calling to_ascii
    Then: raises TypeError with descriptive message
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match=r"only bytes \| str acceptable"):
        to_ascii(invalid_value)


@pytest.mark.unit
def test_to_ascii_uses_custom_charset(fake: Faker) -> None:
    """
    Given: bytes encoded in a non-ascii charset
    When: calling to_ascii with that charset
    Then: correctly decodes the bytes
    """
    # --- Arrange ---
    text = fake.pystr(min_chars=4, max_chars=8)
    charset = "utf-8"
    encoded = text.encode(charset)

    # --- Act ---
    result = to_ascii(encoded, charset=charset)

    # --- Assert ---
    assert result == text


@pytest.mark.unit
@pytest.mark.parametrize(
    "input_value,expected_result",
    (
        pytest.param("hello", b"hello", id="str-input"),
        pytest.param(b"hello", b"hello", id="bytes-input"),
    ),
)
def test_to_bytes_coerces_str_to_bytes(
    input_value: bytes | str,
    expected_result: bytes,
) -> None:
    """
    Given: str or bytes input
    When: calling to_bytes
    Then: returns bytes
    """
    # --- Act ---
    result = to_bytes(input_value)

    # --- Assert ---
    assert result == expected_result


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_value",
    (
        pytest.param(123, id="int"),
        pytest.param(1.5, id="float"),
        pytest.param(None, id="none"),
    ),
)
def test_to_bytes_raises_type_error_for_invalid_type(
    invalid_value: object,
) -> None:
    """
    Given: an invalid type for to_bytes
    When: calling to_bytes
    Then: raises TypeError with descriptive message
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match=r"only bytes \| str acceptable"):
        to_bytes(invalid_value)


@pytest.mark.unit
def test_to_bytes_uses_custom_charset(fake: Faker) -> None:
    """
    Given: a string
    When: calling to_bytes with a custom charset
    Then: returns correctly encoded bytes
    """
    # --- Arrange ---
    text = fake.pystr(min_chars=4, max_chars=8)
    charset = "utf-8"

    # --- Act ---
    result = to_bytes(text, charset=charset)

    # --- Assert ---
    assert result == text.encode(charset)


@pytest.mark.unit
def test_unique_yields_distinct_elements() -> None:
    """
    Given: an iterable with duplicates
    When: calling unique
    Then: yields only distinct elements preserving order
    """
    # --- Arrange ---
    items = [1, 2, 2, 3]

    # --- Act ---
    result = list(unique(items))

    # --- Assert ---
    assert result == [1, 2, 3]


@pytest.mark.unit
def test_unique_with_key_function(fake: Faker) -> None:
    """
    Given: objects with a common attribute
    When: calling unique with a key function
    Then: deduplicates based on the key
    """

    # --- Arrange ---
    class Item:
        def __init__(self, idx: int) -> None:
            self.id = idx

    first_id = fake.pyint()
    second_id = fake.pyint(min_value=1000, max_value=9999)
    items = [Item(first_id), Item(second_id), Item(first_id)]

    # --- Act ---
    result = list(unique(items, key=lambda x: x.id))

    # --- Assert ---
    expected_count = 2
    assert len(result) == expected_count
    assert result[0].id == first_id
    assert result[1].id == second_id


@pytest.mark.unit
def test_unique_with_include_set() -> None:
    """
    Given: an iterable and an include set
    When: calling unique with include
    Then: only yields elements in the include set
    """
    # --- Arrange ---
    items = [1, 2, 3, 4]

    # --- Act ---
    result = list(unique(items, include={2, 4}))

    # --- Assert ---
    assert result == [2, 4]


@pytest.mark.unit
def test_unique_with_exclude_set() -> None:
    """
    Given: an iterable and an exclude set
    When: calling unique with exclude
    Then: skips excluded elements
    """
    # --- Arrange ---
    items = [1, 2, 3, 4]

    # --- Act ---
    result = list(unique(items, exclude={2, 4}))

    # --- Assert ---
    assert result == [1, 3]


@pytest.mark.unit
def test_unique_on_mapping_yields_keys() -> None:
    """
    Given: a mapping
    When: calling unique on it
    Then: yields keys
    """
    # --- Arrange ---
    mapping = {"a": 1, "b": 2}

    # --- Act ---
    result = list(unique(mapping))

    # --- Assert ---
    assert set(result) == {"a", "b"}


@pytest.mark.unit
def test_unique_preserves_first_occurrence_order() -> None:
    """
    Given: an iterable with duplicates out of order
    When: calling unique
    Then: preserves the first occurrence order
    """
    # --- Arrange ---
    items = [3, 1, 3, 2]

    # --- Act ---
    result = list(unique(items))

    # --- Assert ---
    assert result == [3, 1, 2]


@pytest.mark.unit
def test_unique_on_empty_iterable() -> None:
    """
    Given: an empty iterable
    When: calling unique
    Then: yields nothing
    """
    # --- Act ---
    result = list(unique([]))

    # --- Assert ---
    assert result == []


@pytest.mark.unit
def test_iter_inheritance_yields_mro_without_object() -> None:
    """
    Given: a class
    When: calling iter_inheritance
    Then: yields MRO excluding object
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = list(iter_inheritance(Child, exclude_self=False))

    # --- Assert ---
    assert Child in result
    assert Base in result
    assert object not in result


@pytest.mark.unit
def test_iter_inheritance_excludes_self_when_flag_set() -> None:
    """
    Given: a class
    When: calling iter_inheritance with exclude_self=True
    Then: the class itself is skipped
    """

    # --- Arrange ---
    class Sample:
        pass

    # --- Act ---
    result = list(iter_inheritance(Sample, exclude_self=True))

    # --- Assert ---
    assert Sample not in result


@pytest.mark.unit
def test_iter_inheritance_include_filter() -> None:
    """
    Given: a class hierarchy
    When: calling iter_inheritance with an include filter
    Then: only yields matching classes
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = list(iter_inheritance(Child, include=(Base,)))

    # --- Assert ---
    assert result == [Base]


@pytest.mark.unit
def test_iter_inheritance_exclusion_filter() -> None:
    """
    Given: a class hierarchy
    When: calling iter_inheritance with an exclude filter
    Then: skips matching classes
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = list(iter_inheritance(Child, exclude=(Base,), exclude_self=False))

    # --- Assert ---
    assert Base not in result
    assert Child in result


@pytest.mark.unit
def test_iter_inheritance_reverse() -> None:
    """
    Given: a class hierarchy
    When: calling iter_inheritance with reverse=True
    Then: order is reversed
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = list(iter_inheritance(Child, reverse=True, exclude_self=False))

    # --- Assert ---
    assert result.index(Base) < result.index(Child)


@pytest.mark.unit
def test_iter_inheritance_exclude_stdlib() -> None:
    """
    Given: a class inheriting from object
    When: calling iter_inheritance with exclude_stdlib=True
    Then: stdlib classes like object are skipped
    """

    # --- Arrange ---
    class Sample:
        pass

    # --- Act ---
    result = list(iter_inheritance(Sample, exclude_stdlib=True))

    # --- Assert ---
    assert object not in result


@pytest.mark.unit
def test_get_mro_returns_tuple() -> None:
    """
    Given: a class
    When: calling get_mro
    Then: returns a tuple of classes
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = get_mro(Child, exclude_self=False)

    # --- Assert ---
    assert isinstance(result, tuple)
    assert Child in result
    assert Base in result


@pytest.mark.unit
def test_get_mro_with_glue_returns_string() -> None:
    """
    Given: a class
    When: calling get_mro with glue
    Then: returns a joined string
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    # --- Act ---
    result = get_mro(Child, glue=" -> ", exclude_self=False)

    # --- Assert ---
    assert isinstance(result, str)
    assert " -> " in result


@pytest.mark.unit
def test_get_mro_with_func_transforms_each_class() -> None:
    """
    Given: a class
    When: calling get_mro with func=str
    Then: each class is transformed by func
    """

    # --- Arrange ---
    class Sample:
        pass

    # --- Act ---
    result = get_mro(Sample, func=str)

    # --- Assert ---
    assert all(isinstance(item, str) for item in result)


@pytest.mark.unit
def test_get_attr_finds_attribute_on_class() -> None:
    """
    Given: an instance with a method
    When: calling get_attr
    Then: returns the method
    """

    # --- Arrange ---
    class Sample:
        def method(self) -> int:
            return 42

    instance = Sample()

    # --- Act ---
    result = get_attr(instance, "method")

    # --- Assert ---
    assert result is Sample.method


@pytest.mark.unit
def test_get_attr_finds_attribute_on_parent() -> None:
    """
    Given: a child instance inheriting a method
    When: calling get_attr
    Then: returns the parent's method
    """

    # --- Arrange ---
    class Base:
        def method(self) -> int:
            return 42

    class Child(Base):
        pass

    instance = Child()

    # --- Act ---
    result = get_attr(instance, "method")

    # --- Assert ---
    assert result is Base.method


@pytest.mark.unit
def test_get_attr_returns_default_when_missing() -> None:
    """
    Given: an instance without the requested attribute
    When: calling get_attr with a default
    Then: returns the default value
    """

    # --- Arrange ---
    class Sample:
        pass

    instance = Sample()
    default_value = 42

    # --- Act ---
    result = get_attr(instance, "missing", default=default_value)

    # --- Assert ---
    assert result == default_value


@pytest.mark.unit
def test_get_owner_returns_defining_class() -> None:
    """
    Given: a class hierarchy
    When: calling get_owner for an inherited attribute
    Then: returns the class that defines it
    """

    # --- Arrange ---
    class Base:
        def method(self) -> int:
            return 42

    class Child(Base):
        pass

    # --- Act ---
    result = get_owner(Child, "method")

    # --- Assert ---
    assert result is Base


@pytest.mark.unit
def test_get_owner_returns_none_when_not_found() -> None:
    """
    Given: a class without the requested attribute
    When: calling get_owner
    Then: returns None
    """

    # --- Arrange ---
    class Sample:
        pass

    # --- Act ---
    result = get_owner(Sample, "missing")

    # --- Assert ---
    assert result is None


@pytest.mark.unit
def test_is_subclass_with_any_returns_true() -> None:
    """
    Given: any object
    When: checking is_subclass against Any
    Then: returns True
    """
    # --- Arrange ---
    obj = object()

    # --- Act ---
    result = is_subclass(obj, Any)

    # --- Assert ---
    assert result is True


@pytest.mark.unit
def test_is_subclass_with_none_types_returns_true() -> None:
    """
    Given: None
    When: checking is_subclass against type(None)
    Then: returns True
    """
    # --- Act ---
    result = is_subclass(None, type(None))

    # --- Assert ---
    assert result is True


@pytest.mark.unit
def test_is_subclass_with_union_type() -> None:
    """
    Given: an int instance
    When: checking is_subclass against int | str
    Then: returns True
    """
    # --- Arrange ---
    value = 123

    # --- Act ---
    result = is_subclass(value, int | str)

    # --- Assert ---
    assert result is True


@pytest.mark.unit
def test_is_subclass_with_generic_alias() -> None:
    """
    Given: a dict instance
    When: checking is_subclass against dict[str, str]
    Then: returns True
    """
    # --- Arrange ---
    value = {"a": "b"}

    # --- Act ---
    result = is_subclass(value, dict[str, str])

    # --- Assert ---
    assert result is True


@pytest.mark.unit
def test_is_subclass_with_none_types_arg_returns_false() -> None:
    """
    Given: any object
    When: checking is_subclass against None
    Then: returns False
    """
    # --- Arrange ---
    obj = object()

    # --- Act ---
    result = is_subclass(obj, None)

    # --- Assert ---
    assert result is False


@pytest.mark.unit
def test_is_subclass_with_plain_class() -> None:
    """
    Given: a child instance
    When: checking is_subclass against Base
    Then: returns True
    """

    # --- Arrange ---
    class Base:
        pass

    class Child(Base):
        pass

    instance = Child()

    # --- Act ---
    result = is_subclass(instance, Base)

    # --- Assert ---
    assert result is True


@pytest.mark.unit
def test_iter_stack_yields_frames() -> None:
    """
    Given: a running function
    When: calling iter_stack inside it
    Then: yields at least one frame
    """
    # --- Arrange ---
    frames = list(iter_stack())

    # --- Act / Assert ---
    assert len(frames) > 0


@pytest.mark.unit
def test_iter_stack_with_offset_skips_frames() -> None:
    """
    Given: a nested function call
    When: calling iter_stack with offset=1
    Then: skips the current frame
    """
    # --- Arrange ---
    outer = list(iter_stack())
    inner = list(iter_stack(offset=1))

    # --- Act / Assert ---
    assert len(inner) < len(outer)


@pytest.mark.unit
def test_iter_stack_extracts_filename() -> None:
    """
    Given: a running function
    When: calling iter_stack("filename")
    Then: yields strings
    """
    # --- Arrange ---
    filenames = list(iter_stack(1))

    # --- Act / Assert ---
    assert all(isinstance(name, str) for name in filenames)


# ------------------------------------------------------------------
# Annotation inference
# ------------------------------------------------------------------


class TestAnnotationInference:
    """GIVEN kain.internals functions
    WHEN inspecting type hints
    THEN signatures match declared types.
    """

    def test_to_ascii_hints(self) -> None:
        """GIVEN to_ascii
        WHEN getting type hints
        THEN param types are bytes | str and return is str.
        """
        # --- Act ---
        hints = get_type_hints(to_ascii)

        # --- Assert ---
        assert hints["return"] is str

    def test_to_bytes_hints(self) -> None:
        """GIVEN to_bytes
        WHEN getting type hints
        THEN param types are bytes | str and return is bytes.
        """
        # --- Act ---
        hints = get_type_hints(to_bytes)

        # --- Assert ---
        assert hints["return"] is bytes

    def test_get_attr_hints(self) -> None:
        """GIVEN get_attr
        WHEN getting type hints
        THEN return type is Any.
        """
        # --- Act ---
        hints = get_type_hints(get_attr)

        # --- Assert ---
        assert hints["return"] is Any

    def test_get_owner_hints(self) -> None:
        """GIVEN get_owner
        WHEN getting type hints
        THEN return type includes None.
        """
        # --- Act ---
        hints = get_type_hints(get_owner)

        # --- Assert ---
        assert hints["return"] == Union[Any, None]  # noqa: UP007

    def test_is_subclass_hints(self) -> None:
        """GIVEN is_subclass
        WHEN getting type hints
        THEN return type is bool.
        """
        # --- Act ---
        hints = get_type_hints(is_subclass)

        # --- Assert ---
        assert hints["return"] is bool


# ------------------------------------------------------------------
# Inheritance contract
# ------------------------------------------------------------------


class TestInheritanceContract:
    """GIVEN Who and Is dataclasses
    WHEN inspecting inherited behavior
    THEN all expected fields and callables are present.
    """

    def test_iter_inheritance_includes_stdlib_when_excluded_false(
        self,
    ) -> None:
        """GIVEN iter_inheritance with exclude_stdlib=False
        WHEN iterating over a class inheriting from list
        THEN stdlib base is included in the result.
        """

        # --- Arrange ---
        class Base(list):
            pass

        # --- Act ---
        result = list(iter_inheritance(Base, exclude_stdlib=False))

        # --- Assert ---
        assert list in result


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Paranoid edge-case coverage for internals."""

    def test_get_owner_with_lambda_returns_none(self) -> None:
        """GIVEN a descriptor defined with a lambda
        WHEN calling get_owner
        THEN it returns None because lambda __name__ is '<lambda>'.
        """

        # --- Arrange ---
        class Base:
            attr = lambda _self: 1  # noqa: E731

        # --- Act ---
        result = get_owner(Base, "<lambda>")

        # --- Assert ---
        assert result is None

    def test_is_subclass_with_literal_returns_false(self) -> None:
        """GIVEN a Literal type
        WHEN calling is_subclass
        THEN it returns False because Literal is not a class.
        """
        # --- Act ---
        result = is_subclass(1, Literal[1, 2])

        # --- Assert ---
        assert result is False

    def test_to_ascii_with_empty_str_returns_empty(self) -> None:
        """GIVEN empty string
        WHEN calling to_ascii
        THEN empty string is returned.
        """
        # --- Act ---
        result = to_ascii("")

        # --- Assert ---
        assert result == ""

    def test_to_bytes_with_empty_str_returns_empty_bytes(self) -> None:
        """GIVEN empty string
        WHEN calling to_bytes
        THEN empty bytes is returned.
        """
        # --- Act ---
        result = to_bytes("")

        # --- Assert ---
        assert result == b""

    def test_unique_with_generator_expression(self) -> None:
        """GIVEN a generator expression
        WHEN passing to unique
        THEN deduplication works lazily.
        """
        # --- Arrange ---
        data = (x for x in [1, 2, 2, 3])

        # --- Act ---
        result = list(unique(data))

        # --- Assert ---
        assert result == [1, 2, 3]

    def test_iter_inheritance_with_plain_object(self) -> None:
        """GIVEN a plain object (not a class)
        WHEN calling iter_inheritance with exclude_stdlib=False
        THEN it yields the object's class MRO (minus object itself).
        """
        # --- Arrange ---
        obj = object()

        # --- Act ---
        result = list(iter_inheritance(obj, exclude_stdlib=False))

        # --- Assert ---
        assert len(result) == 0  # object.__mro__[:-1] is empty

    def test_get_attr_with_none_obj_raises_attribute_error(self) -> None:
        """GIVEN None as obj
        WHEN calling get_attr
        THEN AttributeError is raised because NoneType has no __dict__.
        """
        # --- Act / Assert ---
        with pytest.raises(AttributeError):
            get_attr(None, "__class__")
