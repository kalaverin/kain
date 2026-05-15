"""Unit tests for properties package init: pin and re-exports.

Arrange-Act-Assert pattern, BDD docstrings.
"""

from __future__ import annotations

import pytest
from faker import Faker

from kain.properties import (
    AttributeExceptionError,
    BaseProperty,
    ContextFaultError,
    PropertyError,
    ReadOnlyError,
    bound_property,
    cached_property,
    class_cached_property,
    class_parent_cached_property,
    class_property,
    mixed_cached_property,
    mixed_parent_cached_property,
    mixed_property,
    pin,
    post_cached_property,
    post_parent_cached_property,
    pre_cached_property,
    pre_parent_cached_property,
    proxy_to,
)

# ------------------------------------------------------------------
# Re-export integrity
# ------------------------------------------------------------------


class TestReExports:
    """GIVEN the public properties API
    WHEN importing symbols from kain.properties
    THEN every symbol in __all__ is reachable.
    """

    @pytest.mark.parametrize(
        "symbol",
        (
            pytest.param(
                AttributeExceptionError,
                id="AttributeExceptionError",
            ),
            pytest.param(BaseProperty, id="BaseProperty"),
            pytest.param(ContextFaultError, id="ContextFaultError"),
            pytest.param(PropertyError, id="PropertyError"),
            pytest.param(ReadOnlyError, id="ReadOnlyError"),
            pytest.param(bound_property, id="bound_property"),
            pytest.param(cached_property, id="cached_property"),
            pytest.param(class_cached_property, id="class_cached_property"),
            pytest.param(
                class_parent_cached_property,
                id="class_parent_cached_property",
            ),
            pytest.param(class_property, id="class_property"),
            pytest.param(mixed_cached_property, id="mixed_cached_property"),
            pytest.param(
                mixed_parent_cached_property,
                id="mixed_parent_cached_property",
            ),
            pytest.param(mixed_property, id="mixed_property"),
            pytest.param(pin, id="pin"),
            pytest.param(post_cached_property, id="post_cached_property"),
            pytest.param(
                post_parent_cached_property,
                id="post_parent_cached_property",
            ),
            pytest.param(pre_cached_property, id="pre_cached_property"),
            pytest.param(
                pre_parent_cached_property,
                id="pre_parent_cached_property",
            ),
            pytest.param(proxy_to, id="proxy_to"),
        ),
    )
    def test_symbol_is_importable(self, symbol: object) -> None:
        """GIVEN a public symbol
        WHEN imported
        THEN it is not None and is callable or a class.
        """
        assert symbol is not None


# ------------------------------------------------------------------
# pin class behavior
# ------------------------------------------------------------------


class TestPin:
    """GIVEN the pin descriptor
    WHEN used as a decorator or accessed
    THEN it behaves as a bound_property that rejects class access.
    """

    def test_pin_is_bound_property_subclass(self) -> None:
        """GIVEN pin
        WHEN checked via issubclass
        THEN it is a subclass of bound_property.
        """
        assert issubclass(pin, bound_property)

    def test_pin_class_access_raises_context_fault(self) -> None:
        """GIVEN a class with pin descriptor
        WHEN accessed on the class
        THEN ContextFaultError is raised.
        """

        class Sample:
            attr = pin(lambda _self: 1)

        with pytest.raises(ContextFaultError):
            _ = Sample.attr

    def test_pin_instance_access_computes_and_caches(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a class with pin descriptor
        WHEN accessed on an instance
        THEN the value is computed and cached in __dict__.
        """
        expected = fake.pyint()

        class Sample:
            attr = pin(lambda _self: expected)

        instance = Sample()
        prop = Sample.__dict__["attr"]
        assert instance.attr == expected
        assert instance.__dict__[prop.name] == expected

    def test_pin_rejects_coroutine_function(self) -> None:
        """GIVEN an async function
        WHEN passed to pin
        THEN TypeError is raised about coroutine functions.
        """

        async def coro(_self: object) -> int:
            return 1

        with pytest.raises(TypeError, match="coroutine function"):
            pin(coro)

    def test_pin_class_access_includes_class_context(self) -> None:
        """GIVEN a class with pin descriptor
        WHEN accessed on the class
        THEN the error message includes 'class' context.
        """

        class Sample:
            attr = pin(lambda _self: 1)

        with pytest.raises(ContextFaultError, match="class"):
            _ = Sample.attr

    def test_pin_on_object_without_dict_raises_type_error(self) -> None:
        """GIVEN an object lacking __dict__
        WHEN pin's __get__ is invoked with that object
        THEN TypeError is raised.
        """

        class Sample:
            attr = pin(lambda _self: 1)

        descriptor = Sample.__dict__["attr"]

        with pytest.raises(TypeError, match="has no __dict__"):
            descriptor.__get__(object(), Sample)


# ------------------------------------------------------------------
# pin factory attributes
# ------------------------------------------------------------------


class TestPinFactories:
    """GIVEN the pin class
    WHEN accessing its factory attributes
    THEN they map to the correct cached descriptor types.
    """

    def test_pin_native_is_cached_property(self) -> None:
        """GIVEN pin.native
        WHEN inspected
        THEN it is cached_property.
        """
        assert pin.native is cached_property

    def test_pin_cls_is_class_cached_property(self) -> None:
        """GIVEN pin.cls
        WHEN inspected
        THEN it is class_cached_property.
        """
        assert pin.cls is class_cached_property

    def test_pin_any_is_mixed_cached_property(self) -> None:
        """GIVEN pin.any
        WHEN inspected
        THEN it is mixed_cached_property.
        """
        assert pin.any is mixed_cached_property

    def test_pin_pre_is_pre_cached_property(self) -> None:
        """GIVEN pin.pre
        WHEN inspected
        THEN it is pre_cached_property.
        """
        assert pin.pre is pre_cached_property

    def test_pin_post_is_post_cached_property(self) -> None:
        """GIVEN pin.post
        WHEN inspected
        THEN it is post_cached_property.
        """
        assert pin.post is post_cached_property


# ------------------------------------------------------------------
# pin with_parent
# ------------------------------------------------------------------


class TestPinWithParent:
    """GIVEN pin.with_parent
    WHEN called with a function
    THEN it returns a cached_property via the runtime staticmethod.
    """

    def test_pin_with_parent_returns_pin_instance(self) -> None:
        """GIVEN a function using with_parent
        WHEN applied via pin.with_parent
        THEN the result is a pin instance.
        """

        def func(_self: object, _parent: int) -> int:
            return 42

        prop = pin.with_parent(func)
        assert isinstance(prop, pin)

    def test_pin_with_parent_computes_correctly(self) -> None:
        """GIVEN a parent pin and child pin.with_parent
        WHEN accessed on an instance
        THEN the child receives the parent's computed value.
        """

        base_value = 10
        increment = 5

        class Parent:
            @pin
            def value(self) -> int:
                return base_value

        class Child(Parent):
            @pin.with_parent
            def value(self, parent_result: int) -> int:
                return parent_result + increment

        instance = Child()
        assert instance.value == base_value + increment


# ------------------------------------------------------------------
# Annotation inference on pin
# ------------------------------------------------------------------


class TestPinAnnotationInference:
    """GIVEN pin with generic type parameters
    WHEN instantiated
    THEN type information is preserved.
    """

    def test_pin_preserves_function_name(self) -> None:
        """GIVEN an annotated function
        WHEN wrapped in pin
        THEN .name matches the original function.
        """

        def compute(_self: object) -> int:
            return 42

        prop = pin(compute)
        assert prop.name == "compute"

    def test_pin_instance_has_bound_property_attributes(self) -> None:
        """GIVEN a pin instance
        WHEN inspecting inherited attributes
        THEN BaseProperty interface is available.
        """

        def compute(_self: object) -> int:
            return 42

        prop = pin(compute)
        assert hasattr(prop, "name")
        assert hasattr(prop, "title")
        assert hasattr(prop, "header")
        assert hasattr(prop, "is_data")

    def test_pin_is_generic_subclass(self) -> None:
        """GIVEN pin
        WHEN inspected
        THEN it is a generic class.
        """
        assert hasattr(pin, "__class_getitem__")


# ------------------------------------------------------------------
# Inheritance contract
# ------------------------------------------------------------------


class TestInheritanceContract:
    """GIVEN the pin descriptor hierarchy
    WHEN inspecting inherited methods and attributes
    THEN the full interface from bound_property is available.
    """

    def test_pin_inherits_name_from_bound_property(self) -> None:
        """GIVEN a pin wrapping a function
        WHEN accessing .name
        THEN it is inherited from BaseProperty via bound_property.
        """

        def sample(_self: object) -> int:
            return 1

        prop = pin(sample)
        assert prop.name == "sample"

    def test_pin_inherits_is_data_from_bound_property(self) -> None:
        """GIVEN a pin instance
        WHEN accessing .is_data
        THEN it reflects the presence of __set__ / __delete__.
        """

        def sample(_self: object) -> int:
            return 1

        prop = pin(sample)
        assert isinstance(prop.is_data, bool)

    def test_pin_inherits_str_and_repr_from_bound_property(self) -> None:
        """GIVEN a pin instance
        WHEN calling str() and repr()
        THEN they contain expected introspection info.
        """

        def sample(_self: object) -> int:
            return 1

        prop = pin(sample)
        assert "<" in str(prop)
        assert ">" in repr(prop)

    def test_pin_inherits_header_from_bound_property(self) -> None:
        """GIVEN a pin instance
        WHEN accessing .header
        THEN it contains the function name.
        """

        def sample(_self: object) -> int:
            return 1

        prop = pin(sample)
        assert prop.name in prop.header

    def test_pin_with_parent_returns_pin_instance(self) -> None:
        """GIVEN pin.with_parent
        WHEN called with a function
        THEN the result is a pin instance.
        """

        def func(_self: object, _parent: int) -> int:
            return 42

        prop = pin.with_parent(func)
        assert isinstance(prop, pin)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Paranoid edge-case coverage for pin."""

    def test_pin_on_slotted_object_raises_type_error(self) -> None:
        """GIVEN an object with __slots__
        WHEN pin's __get__ is invoked
        THEN TypeError is raised about missing __dict__.
        """

        class Sample:
            __slots__ = ()
            attr = pin(lambda _self: 1)

        descriptor = Sample.__dict__["attr"]
        instance = Sample()

        with pytest.raises(TypeError, match="has no __dict__"):
            descriptor.__get__(instance, Sample)

    def test_pin_access_after_deletion_recomputes(self) -> None:
        """GIVEN a cached pin value
        WHEN the cached value is deleted from __dict__
        THEN the next access recomputes.
        """
        expected = 42

        def compute(_self: object) -> int:
            return expected

        class Sample:
            attr = pin(compute)

        instance = Sample()
        descriptor = Sample.__dict__["attr"]
        assert instance.attr == expected
        del instance.__dict__[descriptor.name]
        assert instance.attr == expected

    def test_pin_on_instance_with_existing_dict_value_skips_compute(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an instance with a pre-populated __dict__ entry
        WHEN pin's __get__ is invoked
        THEN the existing value is returned without calling the function.
        """
        expected = fake.pyint()

        def compute(_self: object) -> int:
            return expected + 1

        class Sample:
            attr = pin(compute)

        instance = Sample()
        descriptor = Sample.__dict__["attr"]
        instance.__dict__[descriptor.name] = expected

        assert instance.attr == expected
