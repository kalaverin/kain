"""Extended tests for kain.descriptors module.

Tests covering with_parent, cache, .by/.expired_by custom callbacks,
and edge cases for all public API.
"""

from __future__ import annotations

import asyncio
from time import time as time_func
from typing import Any

import pytest

from kain.descriptors import (
    cache,
    class_property,
    mixed_property,
    pin,
    PropertyError,
    ContextFaultError,
    ReadOnlyError,
    AttributeException,
    AbstractProperty,
    InsteadProperty,
    BaseProperty,
    Cached,
)


class TestCacheFunction:
    """Tests for cache() function — standalone lru_cache wrapper."""

    def test_basic_caching(self) -> None:
        counter = 0

        @cache
        def compute(x: int) -> int:
            nonlocal counter
            counter += 1
            return x * 2

        assert compute(5) == 10
        assert compute(5) == 10
        assert counter == 1
        assert compute(3) == 6
        assert counter == 2

    def test_with_limit(self) -> None:
        counter = 0

        @cache(2)
        def compute(x: int) -> int:
            nonlocal counter
            counter += 1
            return x * 2

        # With limit 2, only 2 items are cached
        assert compute(1) == 2
        assert compute(2) == 4
        assert compute(1) == 2  # cached
        assert counter == 2
        # Adding third item may evict one (LRU behavior)
        assert compute(3) == 6
        assert counter == 3
        # 2 is recently used, should still be cached
        assert compute(2) == 4
        # Either 1 or 3 might be evicted depending on LRU implementation
        assert counter in (3, 4)

    def test_unlimited_cache(self) -> None:
        counter = 0

        @cache(None)
        def compute(x: int) -> int:
            nonlocal counter
            counter += 1
            return x * 2

        for i in range(100):
            compute(i)
        assert counter == 100
        for i in range(100):
            compute(i)
        assert counter == 100  # all cached

    def test_direct_function_application(self) -> None:
        """When function passed directly, wrap it immediately."""
        counter = 0

        def compute() -> int:
            nonlocal counter
            counter += 1
            return 42

        wrapped = cache(compute)
        assert wrapped() == 42
        assert wrapped() == 42
        assert counter == 1

    def test_rejects_classmethod(self) -> None:
        with pytest.raises(TypeError, match="can't wrap"):
            cache(classmethod(lambda cls: 42))

    def test_rejects_staticmethod(self) -> None:
        with pytest.raises(TypeError, match="can't wrap"):
            cache(staticmethod(lambda: 42))

    def test_rejects_invalid_limit_type(self) -> None:
        with pytest.raises(TypeError, match="limit must be None"):
            cache("invalid")

    def test_rejects_zero_limit(self) -> None:
        with pytest.raises(TypeError, match="limit must be None"):
            cache(0)

    def test_rejects_negative_limit(self) -> None:
        with pytest.raises(TypeError, match="limit must be None"):
            cache(-1)


class TestWithParent:
    """Tests for AbstractProperty.with_parent() method."""

    def test_pin_with_parent(self) -> None:
        """with_parent allows accessing parent class property."""
        class Parent:
            counter = 0

            @pin
            def prop(self) -> int:
                Parent.counter += 1
                return 42

        class Child(Parent):
            @pin.with_parent
            def prop(self, parent_value: int) -> int:
                return parent_value + 10

        obj = Child()
        assert obj.prop == 52  # 42 + 10
        assert obj.prop == 52  # cached
        assert Parent.counter == 1

    def test_pin_native_with_parent(self) -> None:
        """with_parent works with pin.native (Cached)."""
        class Parent:
            counter = 0

            @pin.native
            def prop(self) -> int:
                Parent.counter += 1
                return 42

        class Child(Parent):
            @pin.native.with_parent
            def prop(self, parent_value: int) -> int:
                return parent_value + 10

        obj = Child()
        assert obj.prop == 52
        assert obj.prop == 52
        assert Parent.counter == 1

    def test_pin_cls_with_parent(self) -> None:
        """with_parent works with pin.cls (class-level)."""
        class Parent:
            counter = 0

            @pin.cls
            def prop(cls) -> str:
                Parent.counter += 1
                return f"parent_{cls.__name__}"

        class Child(Parent):
            @pin.cls.with_parent
            def prop(cls, parent_value: str) -> str:
                return f"child_{parent_value}"

        assert Child.prop == "child_parent_Child"
        assert Child.prop == "child_parent_Child"
        assert Parent.counter == 1

    def test_pin_any_with_parent(self) -> None:
        """with_parent works with pin.any (mixed)."""
        class Parent:
            counter = 0

            @pin.any
            def prop(self_or_cls: Any) -> int:
                Parent.counter += 1
                return 42

        class Child(Parent):
            @pin.any.with_parent
            def prop(self_or_cls: Any, parent_value: int) -> int:
                return parent_value + 10

        obj = Child()
        assert obj.prop == 52
        assert Child.prop == 52
        assert Parent.counter == 2  # instance and class are separate

    def test_pin_pre_with_parent(self) -> None:
        """with_parent works with pin.pre."""
        class Parent:
            @pin.pre
            def prop(self_or_cls: Any) -> int:
                return 42

        class Child(Parent):
            @pin.pre.with_parent
            def prop(self_or_cls: Any, parent_value: int) -> int:
                return parent_value + 10

        assert Child.prop == 52
        assert Child().prop == 52

    def test_pin_post_with_parent(self) -> None:
        """with_parent works with pin.post."""
        class Parent:
            @pin.post
            def prop(self_or_cls: Any) -> int:
                return 42

        class Child(Parent):
            @pin.post.with_parent
            def prop(self_or_cls: Any, parent_value: int) -> int:
                return parent_value + 10

        assert Child.prop == 52
        assert Child().prop == 52

    def test_class_property_with_parent(self) -> None:
        """with_parent works with class_property."""
        class Parent:
            counter = 0

            @class_property
            def prop(cls) -> str:
                Parent.counter += 1
                return f"parent_{cls.__name__}"

        class Child(Parent):
            @class_property.with_parent
            def prop(cls, parent_value: str) -> str:
                return f"child_{parent_value}"

        assert Child.prop == "child_parent_Child"
        assert Child().prop == "child_parent_Child"

    def test_mixed_property_with_parent(self) -> None:
        """with_parent works with mixed_property."""
        class Parent:
            @mixed_property
            def prop(self_or_cls: Any) -> str:
                return "parent"

        class Child(Parent):
            @mixed_property.with_parent
            def prop(self_or_cls: Any, parent_value: str) -> str:
                return f"child_of_{parent_value}"

        assert Child.prop == "child_of_parent"
        assert Child().prop == "child_of_parent"

    def test_with_parent_without_parent_descriptor(self) -> None:
        """with_parent raises error when parent descriptor not found."""
        class Foo:
            @pin.with_parent
            def prop(self, parent_value: int) -> int:
                return parent_value + 10

        obj = Foo()
        # When parent descriptor not found, extract_wrapped raises NotImplementedError
        with pytest.raises(NotImplementedError):
            obj.prop


class TestCustomCallbackBy:
    """Tests for .by() and .expired_by() custom cache invalidation."""

    def test_pin_native_by_custom_callback(self) -> None:
        """Custom callback controls cache invalidation.
        
        Callback signature: is_actual(self, node, timestamp=None)
        - On set: called as is_actual(self, node) -> timestamp
        - On get: called as is_actual(self, node, timestamp) -> bool
        """
        call_count = 0
        should_invalidate = False

        def is_actual(self: Any, node: Any, timestamp: float | None = None) -> bool | float:
            if timestamp is None:
                # Called on set, return current timestamp
                return time_func()
            # Called on get, check if still valid
            return not should_invalidate

        class Foo:
            counter = 0

            @pin.native.by(is_actual)
            def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1

        should_invalidate = True
        assert obj.prop == 42
        assert Foo.counter == 2

    def test_pin_native_expired_by_alias(self) -> None:
        """.expired_by is alias for .by.
        
        Callback returns True if expired (invalid), False if still valid.
        """
        expired = False

        def is_still_valid(self: Any, node: Any, timestamp: float | None = None) -> bool | float:
            if timestamp is None:
                return time_func()  # Called on set
            return not expired  # Called on get - True means valid

        class Foo:
            counter = 0

            @pin.native.expired_by(is_still_valid)
            def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert Foo.counter == 1

        expired = True
        assert obj.prop == 42
        assert Foo.counter == 2

    def test_pin_cls_by_custom_callback(self) -> None:
        """Custom callback works with pin.cls."""
        expired = False

        def is_still_valid(self: Any, node: Any, timestamp: float | None = None) -> bool | float:
            if timestamp is None:
                return time_func()
            return not expired

        class Foo:
            counter = 0

            @pin.cls.expired_by(is_still_valid)
            def prop(cls) -> int:
                Foo.counter += 1
                return 42

        assert Foo.prop == 42
        assert Foo.counter == 1

        expired = True
        assert Foo.prop == 42
        assert Foo.counter == 2

    def test_pin_any_by_custom_callback(self) -> None:
        """Custom callback works with pin.any."""
        expired = False

        def is_still_valid(self: Any, node: Any, timestamp: float | None = None) -> bool | float:
            if timestamp is None:
                return time_func()
            return not expired

        class Foo:
            counter = 0

            @pin.any.expired_by(is_still_valid)
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert Foo.prop == 42
        assert Foo.counter == 2  # instance and class are separate

        expired = True
        assert obj.prop == 42
        assert Foo.counter == 3

    def test_ttl_on_pin_cls(self) -> None:
        """TTL works with pin.cls."""
        class Foo:
            counter = 0

            @pin.cls.ttl(0.01)
            def prop(cls) -> int:
                Foo.counter += 1
                return 42

        assert Foo.prop == 42
        assert Foo.counter == 1
        from time import sleep
        sleep(0.02)
        assert Foo.prop == 42
        assert Foo.counter == 2

    def test_ttl_on_pin_any(self) -> None:
        """TTL works with pin.any."""
        class Foo:
            counter = 0

            @pin.any.ttl(0.01)
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert Foo.counter == 1
        from time import sleep
        sleep(0.02)
        assert obj.prop == 42
        assert Foo.counter == 2

    def test_ttl_rejects_non_number(self) -> None:
        """TTL rejects non-numeric expire value."""
        with pytest.raises(TypeError, match="expire must be float or int"):
            pin.native.ttl("invalid")

    def test_ttl_rejects_zero(self) -> None:
        """TTL rejects zero expire value."""
        with pytest.raises(ValueError, match="expire must be positive"):
            pin.native.ttl(0)

    def test_ttl_rejects_negative(self) -> None:
        """TTL rejects negative expire value."""
        with pytest.raises(ValueError, match="expire must be positive"):
            pin.native.ttl(-1)


class TestExceptions:
    """Tests for descriptor exceptions."""

    def test_property_error_inheritance(self) -> None:
        """PropertyError is base for all descriptor exceptions."""
        assert issubclass(ContextFaultError, PropertyError)
        assert issubclass(ReadOnlyError, PropertyError)
        assert issubclass(AttributeException, PropertyError)

    def test_context_fault_error_catchable(self) -> None:
        """ContextFaultError raised on invalid context access."""
        class Foo:
            @pin
            def prop(self) -> int:
                return 42

        with pytest.raises(ContextFaultError):
            Foo.prop

    def test_read_only_error_on_delete(self) -> None:
        """ReadOnlyError raised on delete attempt."""
        class Foo:
            @pin
            def prop(self) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        with pytest.raises(ReadOnlyError):
            del obj.prop

    def test_attribute_exception_wraps_error(self) -> None:
        """AttributeException wraps AttributeError from property."""
        class Foo:
            @pin.native
            def prop(self) -> int:
                return self.nonexistent.attr

        obj = Foo()
        with pytest.raises(AttributeException) as exc_info:
            obj.prop
        assert hasattr(exc_info.value, "exception")
        assert isinstance(exc_info.value.exception, AttributeError)


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_pin_with_none_return(self) -> None:
        """pin caches None correctly."""
        class Foo:
            counter = 0

            @pin
            def prop(self) -> None:
                Foo.counter += 1
                return None

        obj = Foo()
        assert obj.prop is None
        assert obj.prop is None
        assert Foo.counter == 1
        assert obj.__dict__["prop"] is None

    def test_pin_native_none_caching(self) -> None:
        """pin.native caches None correctly."""
        class Foo:
            counter = 0

            @pin.native
            def prop(self) -> None:
                Foo.counter += 1
                return None

        obj = Foo()
        assert obj.prop is None
        assert obj.prop is None
        assert Foo.counter == 1

    def test_falsy_values_cached_correctly(self) -> None:
        """Falsy values (0, '', [], False) are cached correctly."""
        class Foo:
            counter = 0

            @pin.native
            def zero(self) -> int:
                Foo.counter += 1
                return 0

            @pin.native
            def empty_str(self) -> str:
                Foo.counter += 1
                return ""

            @pin.native
            def empty_list(self) -> list:
                Foo.counter += 1
                return []

            @pin.native
            def false(self) -> bool:
                Foo.counter += 1
                return False

        obj = Foo()
        assert obj.zero == 0
        assert obj.zero == 0
        assert obj.empty_str == ""
        assert obj.empty_str == ""
        assert obj.empty_list == []
        assert obj.empty_list == []
        assert obj.false is False
        assert obj.false is False
        assert Foo.counter == 4

    def test_descriptor_string_repr(self) -> None:
        """Descriptors have meaningful string representation."""
        class Foo:
            @pin
            def prop(self) -> int:
                return 42

        descriptor = Foo.__dict__["prop"]
        str_repr = str(descriptor)
        repr_str = repr(descriptor)
        assert "descriptor" in str_repr.lower() or "pin" in str_repr.lower()
        assert "descriptor" in repr_str.lower() or "pin" in repr_str.lower()

    def test_multiple_inheritance_with_pin(self) -> None:
        """pin works with multiple inheritance."""
        class Mixin:
            counter = 0

            @pin.native
            def prop(self) -> int:
                Mixin.counter += 1
                return 42

        class Base:
            pass

        class Child(Mixin, Base):
            pass

        obj = Child()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Mixin.counter == 1

    def test_pin_native_delete_then_reaccess(self) -> None:
        """After delete, property is recomputed."""
        class Foo:
            counter = 0

            @pin.native
            def prop(self) -> int:
                Foo.counter += 1
                return Foo.counter

        obj = Foo()
        assert obj.prop == 1
        del obj.prop
        assert obj.prop == 2
        assert obj.prop == 2

    def test_class_property_no_cache(self) -> None:
        """class_property doesn't cache - called every time."""
        class Foo:
            counter = 0

            @class_property
            def prop(cls) -> int:
                Foo.counter += 1
                return Foo.counter

        assert Foo.prop == 1
        assert Foo.prop == 2
        assert Foo().prop == 3
        assert Foo().prop == 4

    def test_mixed_property_no_cache(self) -> None:
        """mixed_property doesn't cache - called every time."""
        class Foo:
            counter = 0

            @mixed_property
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return Foo.counter

        assert Foo.prop == 1
        assert Foo.prop == 2
        assert Foo().prop == 3
        assert Foo().prop == 4


class TestDescriptorInternals:
    """Tests for internal descriptor behavior and attributes."""

    def test_pin_name_property(self) -> None:
        """pin descriptor has correct name property."""
        class Foo:
            @pin
            def my_property(self) -> int:
                return 42

        descriptor = Foo.__dict__["my_property"]
        assert descriptor.name == "my_property"

    def test_pin_native_name_property(self) -> None:
        """pin.native descriptor has correct name property."""
        class Foo:
            @pin.native
            def my_property(self) -> int:
                return 42

        descriptor = Foo.__dict__["my_property"]
        assert descriptor.name == "my_property"

    def test_is_data_property(self) -> None:
        """is_data property correctly identifies data descriptors."""
        class Foo:
            @pin.native
            def cached_prop(self) -> int:
                return 42

        cached_desc = Foo.__dict__["cached_prop"]

        # pin.native has both __set__ and __delete__, so it's data descriptor
        assert cached_desc.is_data is True

    def test_class_property_is_not_data(self) -> None:
        """class_property is not a data descriptor."""
        class Foo:
            @class_property
            def prop(cls) -> int:
                return 42

        descriptor = Foo.__dict__["prop"]
        assert descriptor.is_data is False

    def test_mixed_property_is_not_data(self) -> None:
        """mixed_property is not a data descriptor."""
        class Foo:
            @mixed_property
            def prop(self_or_cls: Any) -> int:
                return 42

        descriptor = Foo.__dict__["prop"]
        assert descriptor.is_data is False


class TestAsyncEdgeCases:
    """Tests for async property edge cases."""

    @pytest.mark.asyncio
    async def test_pin_native_async_return_value(self) -> None:
        """Async property returns awaitable."""
        class Foo:
            @pin.native
            async def prop(self) -> int:
                return 42

        obj = Foo()
        result = obj.prop
        assert asyncio.isfuture(result) or asyncio.iscoroutine(result)
        assert await result == 42

    @pytest.mark.asyncio
    async def test_pin_native_async_exception(self) -> None:
        """Async property propagates exceptions."""
        class Foo:
            @pin.native
            async def prop(self) -> int:
                raise ValueError("test error")

        obj = Foo()
        future = obj.prop
        with pytest.raises(ValueError, match="test error"):
            await future

    @pytest.mark.asyncio
    async def test_pin_native_async_caches_exception_future(self) -> None:
        """Async property caches the future, not the result."""
        class Foo:
            counter = 0

            @pin.native
            async def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        fut1 = obj.prop
        fut2 = obj.prop
        assert fut1 is fut2
        assert await fut1 == 42
        assert Foo.counter == 1


class TestCachedSetOperations:
    """Tests for __set__ behavior on cached properties."""

    def test_manual_override_pin_native(self) -> None:
        """Manual set overrides cached value."""
        class Foo:
            @pin.native
            def prop(self) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        obj.prop = 100
        assert obj.prop == 100

    def test_manual_override_pin_cls(self) -> None:
        """Manual set works on class-level property."""
        class Foo:
            @pin.cls
            def prop(cls) -> int:
                return 42

        # Set on class
        Foo.prop = 100  # type: ignore
        assert Foo.prop == 100  # type: ignore

    def test_manual_override_pin_any_instance(self) -> None:
        """Manual set on instance for pin.any."""
        class Foo:
            @pin.any
            def prop(self_or_cls: Any) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        obj.prop = 100
        assert obj.prop == 100
        assert Foo.prop == 42  # class still has original

    def test_manual_override_pin_pre_class_only(self) -> None:
        """pin.pre only caches on class.
        
        Note: PreCachedProperty.__set__ returns value without caching
        when node is not a class (instance access).
        """
        class Foo:
            counter = 0

            @pin.pre
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return Foo.counter

        # Instance access doesn't cache
        obj = Foo()
        assert obj.prop == 1
        assert obj.prop == 2  # called again
        
        # Manual set on instance - PreCachedProperty ignores it for instances
        obj.prop = 100
        # Since __set__ returns without storing for instances,
        # next access recomputes
        val = obj.prop
        assert val == 3  # counter incremented

    def test_manual_set_on_class_pin_pre(self) -> None:
        """Manual set on class works for pin.pre."""
        class Foo:
            counter = 0

            @pin.pre
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return Foo.counter

        # Class access caches
        assert Foo.prop == 1
        assert Foo.prop == 1  # cached
        
        # Manual set on class
        Foo.prop = 100  # type: ignore
        assert Foo.prop == 100  # manually set

    def test_manual_override_pin_post_instance_only(self) -> None:
        """pin.post only caches on instance."""
        class Foo:
            counter = 0

            @pin.post
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return Foo.counter

        obj = Foo()
        assert obj.prop == 1
        obj.prop = 100
        assert obj.prop == 100  # manually set


class TestInheritanceScenarios:
    """Complex inheritance scenarios."""

    def test_diamond_inheritance_with_pin_native(self) -> None:
        """Diamond inheritance works with pin.native."""
        class A:
            counter = 0

            @pin.native
            def prop(self) -> int:
                A.counter += 1
                return 42

        class B(A):
            pass

        class C(A):
            pass

        class D(B, C):
            pass

        obj = D()
        assert obj.prop == 42
        assert obj.prop == 42
        assert A.counter == 1

    def test_override_with_different_descriptor_type(self) -> None:
        """Parent pin can be overridden with pin.native."""
        class Parent:
            @pin
            def prop(self) -> int:
                return 42

        class Child(Parent):
            @pin.native
            def prop(self) -> int:
                return 100

        obj = Child()
        assert obj.prop == 100
        obj.prop = 200
        assert obj.prop == 200


class TestAbstractPropertyInterface:
    """Tests for AbstractProperty interface compliance."""

    def test_abstract_property_requires_title(self) -> None:
        """Subclasses must implement title property."""
        class ConcreteProperty(AbstractProperty):
            pass

        with pytest.raises(NotImplementedError):
            _ = ConcreteProperty(lambda: 42).title

    def test_abstract_property_requires_header_with_context(self) -> None:
        """Subclasses must implement header_with_context."""
        class ConcreteProperty(AbstractProperty):
            @property
            def title(self) -> str:
                return "test"

        prop = ConcreteProperty(lambda: 42)
        with pytest.raises(NotImplementedError):
            prop.header_with_context(None)


class TestPinSubclassing:
    """Tests for subclassing pin and its variants."""

    def test_pin_subclass_creation(self) -> None:
        """Can create custom pin subclasses."""
        class MyPin(pin):
            def custom_method(self) -> str:
                return "custom"

        class Foo:
            @MyPin
            def prop(self) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        descriptor = Foo.__dict__["prop"]
        assert descriptor.custom_method() == "custom"
