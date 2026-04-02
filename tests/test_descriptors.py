"""Tests for kain.descriptors module."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest

from kain.descriptors import (
    class_property,
    mixed_property,
    pin,
    ContextFaultError,
    ReadOnlyError,
)


class TestPin:
    """Tests for @pin (InsteadProperty)."""

    def test_instance_caches_value(self) -> None:
        class Foo:
            counter = 0

            @pin
            def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1
        assert "prop" in obj.__dict__
        assert obj.__dict__["prop"] == 42

    def test_class_access_raises(self) -> None:
        class Foo:
            @pin
            def prop(self) -> int:
                return 42

        with pytest.raises(ContextFaultError):
            Foo.prop

    def test_delete_raises(self) -> None:
        class Foo:
            @pin
            def prop(self) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        with pytest.raises(ReadOnlyError):
            del obj.prop

    def test_rejects_async(self) -> None:
        with pytest.raises(TypeError):
            class Foo:
                @pin
                async def prop(self) -> int:
                    return 42

    def test_instances_have_separate_cache(self) -> None:
        class Foo:
            counter = 0

            @pin
            def prop(self) -> int:
                Foo.counter += 1
                return Foo.counter

        a = Foo()
        b = Foo()
        assert a.prop == 1
        assert b.prop == 2
        assert a.prop == 1
        assert b.prop == 2


class TestPinNative:
    """Tests for @pin.native (Cached)."""

    def test_instance_caches_value(self) -> None:
        class Foo:
            counter = 0

            @pin.native
            def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1
        assert "__instance_memoized__" in obj.__dict__

    def test_manual_set_and_delete(self) -> None:
        class Foo:
            @pin.native
            def prop(self) -> int:
                return 42

        obj = Foo()
        assert obj.prop == 42
        obj.prop = 100
        assert obj.prop == 100
        del obj.prop
        assert obj.prop == 42

    @pytest.mark.asyncio
    async def test_async_support(self) -> None:
        class Foo:
            @pin.native
            async def prop(self) -> int:
                return 42

        obj = Foo()
        result = obj.prop
        assert asyncio.isfuture(result)
        assert await result == 42

    @pytest.mark.asyncio
    async def test_async_caches_future(self) -> None:
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

    def test_ttl_expiration(self) -> None:
        class Foo:
            counter = 0

            @pin.native.ttl(0.01)
            def prop(self) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1
        time.sleep(0.02)
        assert obj.prop == 42
        assert Foo.counter == 2

    def test_instances_have_separate_cache(self) -> None:
        class Foo:
            counter = 0

            @pin.native
            def prop(self) -> int:
                Foo.counter += 1
                return Foo.counter

        a = Foo()
        b = Foo()
        assert a.prop == 1
        assert b.prop == 2
        assert a.prop == 1
        assert b.prop == 2


class TestPinCls:
    """Tests for @pin.cls (child-aware class cached property)."""

    def test_class_access_caches(self) -> None:
        class Foo:
            counter = 0

            @pin.cls
            def prop(cls) -> str:
                Foo.counter += 1
                return cls.__name__

        assert Foo.prop == "Foo"
        assert Foo.prop == "Foo"
        assert Foo.counter == 1
        assert "__class_memoized__" in Foo.__dict__

    def test_instance_access_uses_class(self) -> None:
        class Foo:
            counter = 0

            @pin.cls
            def prop(cls) -> str:
                Foo.counter += 1
                return cls.__name__

        obj = Foo()
        assert obj.prop == "Foo"
        assert obj.prop == "Foo"
        assert Foo.counter == 1

    def test_inheritance_passes_child_class(self) -> None:
        class Foo:
            counter = 0

            @pin.cls
            def prop(cls) -> str:
                Foo.counter += 1
                return cls.__name__

        class Bar(Foo):
            pass

        assert Bar.prop == "Bar"
        assert Bar().prop == "Bar"
        # Both accesses share the same class-level cache for Bar.
        assert Foo.counter == 1
        assert Bar.prop == "Bar"
        assert Bar().prop == "Bar"
        assert Foo.counter == 1

    def test_inheritance_separate_cache_per_class(self) -> None:
        class Foo:
            @pin.cls
            def prop(cls) -> str:
                return cls.__name__

        class Bar(Foo):
            pass

        assert Foo.prop == "Foo"
        assert Bar.prop == "Bar"
        assert Foo.prop == "Foo"


class TestPinAny:
    """Tests for @pin.any (mixed cached property)."""

    def test_instance_caches(self) -> None:
        class Foo:
            counter = 0

            @pin.any
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1
        assert "__instance_memoized__" in obj.__dict__

    def test_class_caches(self) -> None:
        class Foo:
            counter = 0

            @pin.any
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        assert Foo.prop == 42
        assert Foo.prop == 42
        assert Foo.counter == 1
        assert "__class_memoized__" in Foo.__dict__

    def test_instance_and_class_caches_are_separate(self) -> None:
        class Foo:
            counter_i = 0
            counter_c = 0

            @pin.any
            def prop(self_or_cls: Any) -> int:
                if isinstance(self_or_cls, type):
                    Foo.counter_c += 1
                else:
                    Foo.counter_i += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert Foo.prop == 42
        assert obj.prop == 42
        assert Foo.prop == 42
        assert Foo.counter_i == 1
        assert Foo.counter_c == 1

    def test_inheritance_passes_child(self) -> None:
        class Foo:
            @pin.any
            def prop(self_or_cls: Any) -> str:
                if isinstance(self_or_cls, type):
                    return self_or_cls.__name__
                return self_or_cls.__class__.__name__

        class Bar(Foo):
            pass

        assert Bar.prop == "Bar"
        assert Bar().prop == "Bar"


class TestPinPre:
    """Tests for @pin.pre (mixed, caches only on class access)."""

    def test_class_access_caches(self) -> None:
        class Foo:
            counter = 0

            @pin.pre
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        assert Foo.prop == 42
        assert Foo.prop == 42
        assert Foo.counter == 1
        assert "__class_memoized__" in Foo.__dict__

    def test_instance_access_does_not_cache(self) -> None:
        class Foo:
            counter = 0

            @pin.pre
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 2
        cache = obj.__dict__["__instance_memoized__"]
        assert "prop" not in cache

    def test_class_and_instance_are_independent(self) -> None:
        class Foo:
            counter_c = 0
            counter_i = 0

            @pin.pre
            def prop(self_or_cls: Any) -> int:
                if isinstance(self_or_cls, type):
                    Foo.counter_c += 1
                else:
                    Foo.counter_i += 1
                return 42

        obj = Foo()
        assert Foo.prop == 42
        assert Foo.prop == 42
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter_c == 1
        assert Foo.counter_i == 2


class TestPinPost:
    """Tests for @pin.post (mixed, caches only on instance access)."""

    def test_instance_access_caches(self) -> None:
        class Foo:
            counter = 0

            @pin.post
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.counter == 1
        assert "__instance_memoized__" in obj.__dict__

    def test_class_access_does_not_cache(self) -> None:
        class Foo:
            counter = 0

            @pin.post
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return 42

        assert Foo.prop == 42
        assert Foo.prop == 42
        assert Foo.counter == 2
        cache = Foo.__dict__["__class_memoized__"]
        assert "prop" not in cache

    def test_class_and_instance_are_independent(self) -> None:
        class Foo:
            counter_c = 0
            counter_i = 0

            @pin.post
            def prop(self_or_cls: Any) -> int:
                if isinstance(self_or_cls, type):
                    Foo.counter_c += 1
                else:
                    Foo.counter_i += 1
                return 42

        obj = Foo()
        assert obj.prop == 42
        assert obj.prop == 42
        assert Foo.prop == 42
        assert Foo.prop == 42
        assert Foo.counter_i == 1
        assert Foo.counter_c == 2


class TestClassProperty:
    """Tests for @class_property."""

    def test_instance_access_passes_class(self) -> None:
        class Foo:
            @class_property
            def prop(cls) -> str:
                return cls.__name__

        assert Foo().prop == "Foo"

    def test_class_access_passes_actual_class(self) -> None:
        class Foo:
            @class_property
            def prop(cls) -> str:
                return cls.__name__

        class Bar(Foo):
            pass

        # get_node() is dead code for plain class_property,
        # so the actual class (Bar) is passed, not the owner.
        assert Bar.prop == "Bar"
        assert Bar().prop == "Bar"

    def test_no_caching(self) -> None:
        class Foo:
            counter = 0

            @class_property
            def prop(cls) -> int:
                Foo.counter += 1
                return Foo.counter

        assert Foo.prop == 1
        assert Foo.prop == 2
        assert Foo().prop == 3


class TestMixedProperty:
    """Tests for @mixed_property."""

    def test_instance_access_passes_instance(self) -> None:
        class Foo:
            @mixed_property
            def prop(self_or_cls: Any) -> str:
                return (
                    "instance"
                    if not isinstance(self_or_cls, type)
                    else "class"
                )

        assert Foo().prop == "instance"

    def test_class_access_passes_class(self) -> None:
        class Foo:
            @mixed_property
            def prop(self_or_cls: Any) -> str:
                return (
                    "instance"
                    if not isinstance(self_or_cls, type)
                    else "class"
                )

        assert Foo.prop == "class"

    def test_falsy_instance_bug(self) -> None:
        """mixed_property uses `instance or klass`, so falsy instances
        receive the class instead of themselves. This behaviour is
        depended upon by external libraries and must not change.
        """

        class Falsy:
            def __bool__(self) -> bool:
                return False

            @mixed_property
            def prop(self_or_cls: Any) -> Any:
                return self_or_cls

        obj = Falsy()
        assert obj.prop is Falsy

    def test_no_caching(self) -> None:
        class Foo:
            counter = 0

            @mixed_property
            def prop(self_or_cls: Any) -> int:
                Foo.counter += 1
                return Foo.counter

        assert Foo().prop == 1
        assert Foo().prop == 2
        assert Foo.prop == 3
        assert Foo.prop == 4
