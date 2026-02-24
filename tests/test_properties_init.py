"""Tests for the public API exported by ``kain.properties`` and
``kain.properties.cached``.
"""

from __future__ import annotations

import inspect

from kain import properties
from kain.properties import cached
from kain.properties.cached import (
    cached_property,
    class_cached_property,
    class_parent_cached_property,
    mixed_cached_property,
    mixed_parent_cached_property,
    post_cached_property,
    post_parent_cached_property,
    pre_cached_property,
    pre_parent_cached_property,
)


class TestPropertiesAll:
    """Tests for ``kain.properties.__all__`` consistency."""

    def test_all_is_tuple(self) -> None:
        assert isinstance(properties.__all__, tuple)

    def test_all_length(self) -> None:
        assert len(properties.__all__) == 20

    def test_all_has_no_duplicates(self) -> None:
        assert len(properties.__all__) == len(set(properties.__all__))

    def test_all_matches_expected(self) -> None:
        expected = {
            "AttributeException",
            "BaseProperty",
            "ContextFaultError",
            "PropertyError",
            "ReadOnlyError",
            "bound_property",
            "cached_property",
            "class_cached_property",
            "class_parent_cached_property",
            "class_property",
            "mixed_cached_property",
            "mixed_parent_cached_property",
            "mixed_property",
            "pin",
            "post_cached_property",
            "post_parent_cached_property",
            "pre_cached_property",
            "pre_parent_cached_property",
            "proxy_to",
            "cache",
        }
        assert set(properties.__all__) == expected

    def test_all_names_are_accessible(self) -> None:
        for name in properties.__all__:
            assert hasattr(properties, name)


class TestCachedAll:
    """Tests for ``kain.properties.cached.__all__`` consistency."""

    def test_all_is_tuple(self) -> None:
        assert isinstance(cached.__all__, tuple)

    def test_all_length(self) -> None:
        assert len(cached.__all__) == 9

    def test_all_has_no_duplicates(self) -> None:
        assert len(cached.__all__) == len(set(cached.__all__))

    def test_all_matches_expected(self) -> None:
        expected = {
            "cached_property",
            "class_cached_property",
            "class_parent_cached_property",
            "mixed_cached_property",
            "mixed_parent_cached_property",
            "post_cached_property",
            "post_parent_cached_property",
            "pre_cached_property",
            "pre_parent_cached_property",
        }
        assert set(cached.__all__) == expected

    def test_all_names_are_accessible(self) -> None:
        for name in cached.__all__:
            assert hasattr(cached, name)


class TestPropertiesExports:
    """Type and relationship checks for ``kain.properties`` exports."""

    def test_pin_is_class(self) -> None:
        assert inspect.isclass(properties.pin)

    def test_pin_is_bound_property_subclass(self) -> None:
        assert issubclass(properties.pin, properties.bound_property)

    def test_pin_native_is_cached_property(self) -> None:
        assert properties.pin.native is cached_property

    def test_pin_cls_is_class_cached_property(self) -> None:
        assert properties.pin.cls is class_cached_property

    def test_pin_any_is_mixed_cached_property(self) -> None:
        assert properties.pin.any is mixed_cached_property

    def test_pin_pre_is_pre_cached_property(self) -> None:
        assert properties.pin.pre is pre_cached_property

    def test_pin_post_is_post_cached_property(self) -> None:
        assert properties.pin.post is post_cached_property

    def test_class_property_is_class(self) -> None:
        assert inspect.isclass(properties.class_property)

    def test_mixed_property_is_class(self) -> None:
        assert inspect.isclass(properties.mixed_property)

    def test_proxy_to_is_callable(self) -> None:
        assert callable(properties.proxy_to)

    def test_base_property_is_class(self) -> None:
        assert inspect.isclass(properties.BaseProperty)

    def test_bound_property_is_class(self) -> None:
        assert inspect.isclass(properties.bound_property)

    def test_property_error_is_exception(self) -> None:
        assert inspect.isclass(properties.PropertyError)
        assert issubclass(properties.PropertyError, Exception)

    def test_context_fault_error_inheritance(self) -> None:
        assert issubclass(
            properties.ContextFaultError,
            properties.PropertyError,
        )

    def test_read_only_error_inheritance(self) -> None:
        assert issubclass(properties.ReadOnlyError, properties.PropertyError)

    def test_attribute_exception_inheritance(self) -> None:
        assert issubclass(
            properties.AttributeException,
            properties.PropertyError,
        )


class TestPropertiesReExports:
    """Verify that ``kain.properties`` re-exports the same objects as
    ``kain.properties.cached``.
    """

    def test_cached_property_same_object(self) -> None:
        assert properties.cached_property is cached_property

    def test_class_cached_property_same_object(self) -> None:
        assert properties.class_cached_property is class_cached_property

    def test_class_parent_cached_property_same_object(self) -> None:
        assert (
            properties.class_parent_cached_property
            is class_parent_cached_property
        )

    def test_mixed_cached_property_same_object(self) -> None:
        assert properties.mixed_cached_property is mixed_cached_property

    def test_mixed_parent_cached_property_same_object(self) -> None:
        assert (
            properties.mixed_parent_cached_property
            is mixed_parent_cached_property
        )

    def test_pre_cached_property_same_object(self) -> None:
        assert properties.pre_cached_property is pre_cached_property

    def test_pre_parent_cached_property_same_object(self) -> None:
        assert (
            properties.pre_parent_cached_property is pre_parent_cached_property
        )

    def test_post_cached_property_same_object(self) -> None:
        assert properties.post_cached_property is post_cached_property

    def test_post_parent_cached_property_same_object(self) -> None:
        assert (
            properties.post_parent_cached_property
            is post_parent_cached_property
        )


class TestCachedExports:
    """Type checks for ``kain.properties.cached`` exports."""

    def test_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.cached_property)

    def test_class_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.class_cached_property)

    def test_class_parent_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.class_parent_cached_property)

    def test_mixed_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.mixed_cached_property)

    def test_mixed_parent_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.mixed_parent_cached_property)

    def test_pre_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.pre_cached_property)

    def test_pre_parent_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.pre_parent_cached_property)

    def test_post_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.post_cached_property)

    def test_post_parent_cached_property_is_class(self) -> None:
        assert inspect.isclass(cached.post_parent_cached_property)
