"""Tests for ``kain.properties.proxy_to``."""

from __future__ import annotations

from functools import partial
from operator import itemgetter
from typing import Any

import pytest

from kain.classes import Missing, Nothing
from kain.properties import proxy_to


class Inner:
    """Dummy target object for proxy tests."""

    def __init__(self) -> None:
        self.value = 42

    def foo(self) -> str:
        return "foo"

    def bar(self) -> str:
        return "bar"

    @property
    def baz(self) -> int:
        return 99


class TestProxyToBasic:
    """Basic proxy_to behaviour with string pivot and default bind."""

    def test_proxies_methods_via_bound_property(self) -> None:
        @proxy_to("inner", "foo", "bar")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        assert obj.foo() == "foo"
        assert obj.bar() == "bar"

    def test_caches_result_in_instance_dict(self) -> None:
        @proxy_to("inner", "baz")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        # first access computes and stores
        assert obj.baz == 99
        assert "baz" in obj.__dict__
        # subsequent accesses hit the cache
        assert obj.__dict__["baz"] == 99

    def test_tracks_proxy_fields(self) -> None:
        @proxy_to("inner", "foo", "bar")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        assert hasattr(Wrapper, "__proxy_fields__")
        assert Wrapper.__proxy_fields__ == ["bar", "foo"]


class TestProxyToBindModes:
    """Different ``bind`` strategies."""

    def test_bind_none_copies_from_pivot_descriptor(self) -> None:
        class Descriptor:
            def method(self) -> str:
                return "desc"

        @proxy_to("desc", "method", None)
        class Wrapper:
            desc = Descriptor()

        # direct copy → the bound method from the class-level pivot instance
        assert Wrapper.method() == "desc"
        obj = Wrapper()
        assert obj.method() == "desc"

    def test_bind_none_fallback_to_getattr(self) -> None:
        class Descriptor:
            pass

        Descriptor.dynamic = lambda self: "dynamic"  # type: ignore[attr-defined]

        @proxy_to("desc", "dynamic", None)
        class Wrapper:
            desc = Descriptor()

        obj = Wrapper()
        assert obj.dynamic() == "dynamic"

    def test_custom_bind_callable(self) -> None:
        def custom_bind(func):
            def wrapper(node):
                return f"wrap:{func(node)}"
            return wrapper

        @proxy_to("inner", "value", custom_bind)
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        assert obj.value() == "wrap:42"

    def test_last_string_uses_bound_property(self) -> None:
        # when last mapping arg is a string, bind defaults to bound_property
        @proxy_to("inner", "foo")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        assert obj.foo() == "foo"
        assert "foo" in obj.__dict__


class TestProxyToNonStringPivot:
    """Proxying to a non-string (object) pivot."""

    def test_uses_pivot_object_directly(self) -> None:
        target = Inner()

        @proxy_to(target, "foo")
        class Wrapper:
            pass

        obj = Wrapper()
        assert obj.foo() == "foo"

    def test_ignores_instance_when_pivot_is_object(self) -> None:
        target = Inner()

        @proxy_to(target, "value")
        class Wrapper:
            def __init__(self) -> None:
                self.other = 100  # should be ignored

        obj = Wrapper()
        assert obj.value == 42


class TestProxyToSafeMode:
    """``safe`` parameter behaviour."""

    def test_safe_true_blocks_existing_attributes(self) -> None:
        with pytest.raises(TypeError, match="already exists"):
            @proxy_to("inner", "foo")
            class Wrapper:
                def __init__(self) -> None:
                    self.inner = Inner()

                def foo(self) -> str:
                    return "local"

    def test_safe_false_allows_overwrite(self) -> None:
        @proxy_to("inner", "foo", safe=False)
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

            def foo(self) -> str:
                return "local"

        obj = Wrapper()
        # the proxy overwrites the original method
        assert obj.foo() == "foo"

    def test_safe_allows_private_attributes(self) -> None:
        class Target:
            def _private(self) -> str:
                return "proxied"

        @proxy_to("inner", "_private")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Target()

            def _private(self) -> str:
                return "local"

        # _private starts with underscore, so safe check is skipped
        obj = Wrapper()
        assert obj._private() == "proxied"


class TestProxyToDefaults:
    """``default`` parameter behaviour."""

    def test_default_raises_when_pivot_is_none(self) -> None:
        @proxy_to("inner", "foo", default=Nothing)
        class Wrapper:
            def __init__(self) -> None:
                self.inner = None

        obj = Wrapper()
        with pytest.raises(AttributeError, match="is None"):
            obj.foo

    def test_default_returns_fallback_when_pivot_is_none(self) -> None:
        @proxy_to("inner", "foo", default="fallback")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = None

        obj = Wrapper()
        assert obj.foo == "fallback"

    def test_default_raises_when_attr_missing(self) -> None:
        @proxy_to("inner", "missing", default=Nothing)
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        with pytest.raises(AttributeError, match="isn't exists"):
            obj.missing

    def test_default_returns_fallback_when_attr_missing(self) -> None:
        @proxy_to("inner", "missing", default="fallback")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        assert obj.missing == "fallback"


class TestProxyToPre:
    """``pre`` parameter behaviour."""

    def test_pre_wraps_result_in_partial(self) -> None:
        @proxy_to("inner", "value", pre=str)
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        result = obj.value
        assert isinstance(result, partial)
        assert result() == "42"


class TestProxyToGetter:
    """Custom ``getter`` parameter."""

    def test_itemgetter_for_dict_pivot(self) -> None:
        @proxy_to("data", "name", "age", getter=itemgetter)
        class Wrapper:
            def __init__(self) -> None:
                self.data = {"name": "Alice", "age": 30}

        obj = Wrapper()
        assert obj.name == "Alice"
        assert obj.age == 30

    def test_custom_getter_callable(self) -> None:
        def custom_getter(name: str):
            def fetch(entity: Any) -> Any:
                return entity.get(name, "unknown")
            return fetch

        @proxy_to("data", "name", getter=custom_getter)
        class Wrapper:
            def __init__(self) -> None:
                self.data = {"name": "Bob"}

        obj = Wrapper()
        assert obj.name == "Bob"


class TestProxyToValidation:
    """Input validation and error paths."""

    def test_non_class_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="isn't a class"):
            proxy_to("inner", "foo")(lambda x: x)  # type: ignore[arg-type]

    def test_empty_mapping_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            @proxy_to("inner")
            class Wrapper:
                pass

    def test_single_non_string_mapping_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            @proxy_to("inner", 123)
            class Wrapper:
                pass

    def test_missing_pivot_attribute_raises(self) -> None:
        @proxy_to("inner", "foo")
        class Wrapper:
            pass

        obj = Wrapper()
        with pytest.raises(AttributeError, match="isn't exists"):
            obj.foo()


class TestProxyToLogging:
    """Warning logs when default fallback is used."""

    def test_logs_warning_when_pivot_is_none(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        @proxy_to("inner", "foo", default="fallback")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = None

        obj = Wrapper()
        with caplog.at_level(logging.WARNING):
            assert obj.foo == "fallback"

        assert "is None" in caplog.text
        assert "return str" in caplog.text

    def test_logs_warning_when_attr_missing(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        @proxy_to("inner", "missing", default="fallback")
        class Wrapper:
            def __init__(self) -> None:
                self.inner = Inner()

        obj = Wrapper()
        with caplog.at_level(logging.WARNING):
            assert obj.missing == "fallback"

        assert "isn't exists" in caplog.text
        assert "return str" in caplog.text
