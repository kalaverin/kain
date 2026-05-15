"""Unit tests for the proxy_to class decorator.

Arrange-Act-Assert pattern, BDD docstrings.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING

import pytest
from faker import Faker
from pytest_mock import MockerFixture

from kain.properties.primitives import bound_property
from kain.properties.proxy_to import proxy_to

if TYPE_CHECKING:
    from typing import Any


# ------------------------------------------------------------------
# Basic proxy_to with string pivot
# ------------------------------------------------------------------


class TestProxyToStringPivot:
    """GIVEN proxy_to with a string pivot
    WHEN applied to a class
    THEN attributes are forwarded to the pivot object's attributes.
    """

    def test_proxy_to_forwards_attribute_to_pivot(self, fake: Faker) -> None:
        """GIVEN a class with a pivot attribute
        WHEN proxy_to redirects a method to the pivot
        THEN accessing the method returns the pivot's attribute value.
        """
        expected = fake.pystr()

        class Target:
            name = expected

        @proxy_to("target", "name")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        instance = Wrapper()
        assert instance.name == expected

    def test_proxy_to_multiple_methods(self, fake: Faker) -> None:
        """GIVEN a class with multiple pivot attributes
        WHEN proxy_to redirects multiple methods
        THEN each method returns the corresponding pivot value.
        """
        first = fake.pystr()
        second = fake.pyint()

        class Target:
            alpha = first
            beta = second

        @proxy_to("target", "alpha", "beta")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        instance = Wrapper()
        assert instance.alpha == first
        assert instance.beta == second

    def test_proxy_to_populates_proxy_fields(self, fake: Faker) -> None:
        """GIVEN proxy_to applied to a class
        WHEN inspecting __proxy_fields__
        THEN it contains the forwarded method names sorted.
        """

        class Target:
            a = fake.pystr()
            b = fake.pystr()

        @proxy_to("target", "b", "a")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        assert Wrapper.__proxy_fields__ == ["a", "b"]

    def test_proxy_to_appends_to_existing_proxy_fields(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a class already having __proxy_fields__
        WHEN proxy_to is applied again
        THEN new fields are appended and sorted.
        """

        class Target:
            a = fake.pystr()
            b = fake.pystr()
            c = fake.pystr()

        @proxy_to("target", "a")
        @proxy_to("target", "c", "b")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        assert Wrapper.__proxy_fields__ == ["a", "b", "c"]


# ------------------------------------------------------------------
# Object pivot (non-string)
# ------------------------------------------------------------------


class TestProxyToObjectPivot:
    """GIVEN proxy_to with an object pivot
    WHEN applied to a class
    THEN attributes are forwarded directly to the pivot object.
    """

    def test_proxy_to_object_pivot_forwards_attribute(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a module-level object pivot
        WHEN proxy_to redirects a method to it
        THEN accessing the method returns the pivot's attribute.
        """
        expected = fake.pystr()

        class Pivot:
            value = expected

        pivot = Pivot()

        @proxy_to(pivot, "value")
        class Wrapper:
            pass

        assert Wrapper().value == expected

    def test_proxy_to_object_pivot_missing_attribute_raises(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an object pivot without the requested attribute
        WHEN the proxied method is accessed
        THEN AttributeError is raised with a descriptive message.
        """

        class Pivot:
            pass

        pivot = Pivot()

        @proxy_to(pivot, "missing")
        class Wrapper:
            pass

        with pytest.raises(AttributeError, match="does not exist"):
            _ = Wrapper().missing


# ------------------------------------------------------------------
# Safety and collision handling
# ------------------------------------------------------------------


class TestProxyToSafety:
    """GIVEN proxy_to with safe=True (default)
    WHEN a method already exists on the class
    THEN TypeError is raised to prevent accidental override.
    """

    def test_safe_mode_raises_on_existing_public_attribute(self) -> None:
        """GIVEN a class with an existing public method
        WHEN proxy_to tries to override it with safe=True
        THEN TypeError is raised.
        """

        class Target:
            existing = 42

        with pytest.raises(TypeError, match="already exists"):

            @proxy_to("target", "existing")
            class Wrapper:
                target = Target()

                def existing(self) -> int:
                    return 99

    def test_unsafe_mode_allows_override(self, fake: Faker) -> None:
        """GIVEN a class with an existing public method
        WHEN proxy_to overrides it with safe=False
        THEN the proxy replaces the original method.
        """
        expected = fake.pystr()

        class Target:
            existing = expected

        @proxy_to("target", "existing", safe=False)
        class Wrapper:
            target = Target()

            def existing(self) -> int:  # type: ignore[assignment][no-redef]
                return 99

        assert Wrapper().existing == expected

    def test_safe_mode_ignores_private_attributes(self, fake: Faker) -> None:
        """GIVEN a class with an existing private method
        WHEN proxy_to tries to override it with safe=True
        THEN it is allowed because private names start with '_'.
        """
        expected = fake.pystr()

        class Target:
            _private = expected

        @proxy_to("target", "_private")
        class Wrapper:
            target = Target()

            def _private(self) -> int:
                return 99

        assert Wrapper()._private == expected


# ------------------------------------------------------------------
# Default value handling
# ------------------------------------------------------------------


class TestProxyToDefault:
    """GIVEN proxy_to with a default value
    WHEN the pivot attribute is missing or None
    THEN the default is returned instead of raising.
    """

    def test_default_returned_when_pivot_is_none(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a pivot that is None
        WHEN the proxied method is accessed with a default
        THEN the default value is returned.
        """
        default_value = fake.pystr()

        @proxy_to("target", "name", default=default_value)
        class Wrapper:
            def __init__(self) -> None:
                self.target = None

        instance = Wrapper()
        assert instance.name == default_value

    def test_default_returned_when_attribute_missing(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a pivot missing the requested attribute
        WHEN the proxied method is accessed with a default
        THEN the default value is returned.
        """
        default_value = fake.pystr()

        class Target:
            pass

        @proxy_to("target", "missing", default=default_value)
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        instance = Wrapper()
        assert instance.missing == default_value

    def test_no_default_raises_when_pivot_is_none(self) -> None:
        """GIVEN no default and pivot is None
        WHEN the proxied method is accessed
        THEN AttributeError is raised.
        """

        @proxy_to("target", "name")
        class Wrapper:
            def __init__(self) -> None:
                self.target = None

        with pytest.raises(AttributeError):
            _ = Wrapper().name

    def test_default_logs_warning(
        self,
        fake: Faker,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """GIVEN a default value used
        WHEN the proxied method falls back to default
        THEN a warning is logged with context.
        """
        default_value = fake.pystr()

        @proxy_to("target", "name", default=default_value)
        class Wrapper:
            def __init__(self) -> None:
                self.target = None

        with caplog.at_level(
            logging.WARNING,
            logger="kain.properties.proxy_to",
        ):
            _ = Wrapper().name

        assert any("default" in r.message.lower() for r in caplog.records)


# ------------------------------------------------------------------
# Custom getter and pre-processor
# ------------------------------------------------------------------


class TestProxyToGetterAndPre:
    """GIVEN proxy_to with custom getter or pre
    WHEN the proxied method is accessed
    THEN the custom logic is applied.
    """

    def test_custom_getter_is_used(self, fake: Faker) -> None:
        """GIVEN a custom getter function
        WHEN proxy_to uses it
        THEN the getter transforms the lookup.
        """
        expected = fake.pystr()

        class Target:
            data = {"key": expected}  # noqa: RUF012

        def dict_getter(name: str) -> Callable[[object], Any]:
            return lambda obj: obj.data[name]

        @proxy_to("target", "key", getter=dict_getter)
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        assert Wrapper().key == expected

    def test_pre_processor_is_applied(self, fake: Faker) -> None:
        """GIVEN a pre-processor function
        WHEN proxy_to uses it
        THEN the result is wrapped in a partial with pre.
        """
        raw = fake.pystr()

        class Target:
            value = raw

        @proxy_to("target", "value", pre=str.upper)
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        result = Wrapper().value
        assert isinstance(result, partial)
        assert result.func is str.upper
        assert result.args == (raw,)


# ------------------------------------------------------------------
# bind modes
# ------------------------------------------------------------------


class TestProxyToBindModes:
    """GIVEN proxy_to with different bind strategies
    WHEN applied to a class
    THEN the binding mechanism matches the expectation.
    """

    def test_string_last_element_uses_bound_property(self) -> None:
        """GIVEN a string as the last mapping element
        WHEN proxy_to processes it
        THEN bound_property is used as the binding mechanism.
        """

        class Target:
            value = 42

        @proxy_to("target", "value")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        desc = Wrapper.__dict__["value"]
        assert isinstance(desc, bound_property)

    def test_none_last_element_skips_binding(self, fake: Faker) -> None:
        """GIVEN None as the last mapping element
        WHEN proxy_to processes it
        THEN the attribute is set directly without a descriptor.
        """
        expected = fake.pystr()

        class Target:
            value = expected

        @proxy_to("target", "value", None)
        class Wrapper:
            target = Target()

        # Without binding, the value is looked up dynamically
        # through the wrapper closure, not stored as descriptor
        assert Wrapper().value == expected

    def test_explicit_bind_type_is_used(self, fake: Faker) -> None:
        """GIVEN an explicit bind type as the last element
        WHEN proxy_to processes it
        THEN that type is used for binding.
        """
        expected = fake.pystr()

        class Target:
            value = expected

        @proxy_to("target", "value", bound_property)
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        desc = Wrapper.__dict__["value"]
        assert isinstance(desc, bound_property)


# ------------------------------------------------------------------
# Validation errors
# ------------------------------------------------------------------


class TestProxyToValidation:
    """GIVEN invalid inputs to proxy_to
    WHEN applied
    THEN appropriate errors are raised.
    """

    def test_non_class_raises_type_error(self) -> None:
        """GIVEN a non-class object
        WHEN proxy_to is applied to it
        THEN TypeError is raised.
        """
        decorator = proxy_to("pivot", "method")

        with pytest.raises(TypeError, match="isn't a class"):
            decorator(object())

    def test_empty_mapping_list_raises_value_error(self) -> None:
        """GIVEN only a pivot with no methods
        WHEN proxy_to is applied
        THEN ValueError is raised.
        """

        class Target:
            pass

        with pytest.raises(ValueError, match="empty mapping"):

            @proxy_to("target")
            class Wrapper:
                target = Target()


# ------------------------------------------------------------------
# AttributeError messages
# ------------------------------------------------------------------


class TestProxyToErrorMessages:
    """GIVEN proxy_to attribute resolution failures
    WHEN the error is raised
    THEN the message contains actionable context.
    """

    def test_missing_pivot_attribute_error_includes_context(self) -> None:
        """GIVEN a missing pivot attribute
        WHEN accessed
        THEN AttributeError mentions the proxied name and pivot.
        """

        class Target:
            pass

        @proxy_to("target", "missing")
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        with pytest.raises(AttributeError, match="missing") as exc_info:
            _ = Wrapper().missing

        assert "proxied" in str(exc_info.value).lower()

    def test_none_pivot_attribute_error_includes_context(self) -> None:
        """GIVEN a None pivot without default
        WHEN accessed
        THEN AttributeError mentions None context.
        """

        @proxy_to("target", "missing")
        class Wrapper:
            def __init__(self) -> None:
                self.target = None

        with pytest.raises(AttributeError, match="None") as exc_info:
            _ = Wrapper().missing

        assert "proxied" in str(exc_info.value).lower()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------


class TestProxyToLogging:
    """GIVEN proxy_to operations that trigger warnings
    WHEN executed
    THEN structured log entries are emitted.
    """

    def test_fallback_to_default_logs_warning(
        self,
        fake: Faker,
        mocker: MockerFixture,
    ) -> None:
        """GIVEN a missing attribute with a default
        WHEN the proxy falls back
        THEN the logger warning is called with context.
        """
        default_value = fake.pystr()
        mock_logger = mocker.patch("kain.properties.proxy_to.logger")

        class Target:
            pass

        @proxy_to("target", "missing", default=default_value)
        class Wrapper:
            def __init__(self) -> None:
                self.target = Target()

        assert Wrapper().missing == default_value
        mock_logger.warning.assert_called_once()


class TestAnnotationInference:
    """Verify proxy_to preserves type information on decorated classes."""

    def test_proxy_to_returns_class(self) -> None:
        """
        Given: a plain class
        WHEN: decorated with proxy_to
        THEN: the returned object is a class.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        @proxy_to(Pivot(), "method")
        class Target:
            pass

        assert isinstance(Target, type)

    def test_proxy_to_populates_proxy_fields(self) -> None:
        """
        Given: a class decorated with proxy_to
        WHEN: inspecting __proxy_fields__
        THEN: it contains the forwarded method names.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        @proxy_to(Pivot(), "method")
        class Target:
            pass

        assert "method" in Target.__proxy_fields__


class TestInheritanceContract:
    """Verify proxy_to decorated classes preserve behaviour in subclasses."""

    def test_subclass_inherits_proxy_fields(self) -> None:
        """
        Given: a proxy_to decorated base class
        WHEN: a subclass is created
        THEN: __proxy_fields__ is inherited.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        @proxy_to(Pivot(), "method")
        class Base:
            pass

        class Child(Base):
            pass

        assert "method" in Child.__proxy_fields__

    def test_subclass_can_override_proxied_method(self) -> None:
        """
        Given: a proxy_to decorated base class
        WHEN: subclass overrides the proxied method
        THEN: the override takes precedence.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        @proxy_to(Pivot(), "method", safe=False)
        class Base:
            pass

        expected = 2

        class Child(Base):
            def method(self) -> int:
                return expected

        assert Child().method() == expected


class TestEdgeCases:
    """Paranoid edge-case coverage for proxy_to."""

    def test_string_pivot_missing_raises_attribute_error(self) -> None:
        """
        Given: a string pivot that does not exist on the instance
        WHEN: the proxied attribute is accessed
        THEN: AttributeError with diagnostic context is raised.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        @proxy_to("pivot", "method")
        class Target:
            pass

        instance = Target()
        with pytest.raises(AttributeError, match="does not exist"):
            _ = instance.method

    def test_none_pivot_with_nothing_default_raises_attribute_error(
        self,
    ) -> None:
        """
        Given: a pivot that is None and default=Nothing
        WHEN: the proxied attribute is accessed
        THEN: AttributeError is raised.
        """

        class Pivot:
            pass

        @proxy_to(Pivot(), "method")
        class Target:
            pass

        instance = Target()
        with pytest.raises(AttributeError):
            _ = instance.method

    def test_pre_processor_with_string_pivot(self) -> None:
        """
        Given: a pre processor with string pivot
        WHEN: the proxied attribute is accessed
        THEN: a partial(pre, result) is returned.
        """

        class Pivot:
            def method(self) -> int:
                return 1

        def pre(_entity: object, result: int) -> int:
            return result + 1

        @proxy_to("pivot", "method", pre=pre)
        class Target:
            pivot = Pivot()

        instance = Target()
        raw = instance.method
        assert isinstance(raw, partial)
        assert raw.func is pre
