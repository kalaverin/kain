"""Type tests for kain.properties proxy_to and primitives.

Runtime assertions via ``typing.get_type_hints`` plus static usage
patterns that must pass under basedpyright / mypy without errors.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar, assert_type, cast, get_type_hints

from kain.properties.primitives import BaseProperty, bound_property
from kain.properties.proxy_to import proxy_to

T = TypeVar("T")


# ------------------------------------------------------------------
# proxy_to — runtime type hints
# ------------------------------------------------------------------


class TestProxyToRuntimeHints:
    """GIVEN proxy_to function
    WHEN inspecting annotations at runtime
    THEN every public parameter and the return type are declared.
    """

    def test_proxy_to_has_return_annotation(self) -> None:
        hints = get_type_hints(proxy_to)
        assert "return" in hints

    def test_proxy_to_has_getter_annotation(self) -> None:
        hints = get_type_hints(proxy_to)
        assert "getter" in hints

    def test_proxy_to_has_default_annotation(self) -> None:
        hints = get_type_hints(proxy_to)
        assert "default" in hints

    def test_proxy_to_has_pre_annotation(self) -> None:
        hints = get_type_hints(proxy_to)
        assert "pre" in hints

    def test_proxy_to_has_safe_annotation(self) -> None:
        hints = get_type_hints(proxy_to)
        assert "safe" in hints


# ------------------------------------------------------------------
# proxy_to — static inference (verified by type checker)
# ------------------------------------------------------------------


class TestProxyToStaticInference:
    """GIVEN proxy_to decorator factory
    WHEN applied to real classes
    THEN type checkers infer correct signatures.
    """

    def test_basic_string_pivot(self) -> None:
        """GIVEN a string pivot and method name
        WHEN decorating a class
        THEN the class retains its identity.
        """

        @proxy_to("target", "name")
        class Sample:
            target: object = None  # type: ignore[assignment][assignment]

        assert_type(Sample, type[Sample])
        assert "name" in Sample.__dict__
        fields = cast("list[str]", Sample.__proxy_fields__)
        assert fields == ["name"]

    def test_multiple_methods(self) -> None:
        """GIVEN multiple method names
        WHEN decorating a class
        THEN all methods are proxied.
        """

        @proxy_to("target", "alpha", "beta")
        class Sample:
            target: object = None  # type: ignore[assignment][assignment]

        assert_type(Sample, type[Sample])
        fields = cast("list[str]", Sample.__proxy_fields__)
        assert fields == ["alpha", "beta"]

    def test_object_pivot(self) -> None:
        """GIVEN an object pivot
        WHEN decorating a class
        THEN the class is still a type.
        """
        pivot = object()

        @proxy_to(pivot, "value")
        class Sample:
            pass

        assert_type(Sample, type[Sample])

    def test_with_default(self) -> None:
        """GIVEN a default fallback value
        WHEN decorating a class
        THEN the class is still a type.
        """
        fallback: str = "fallback"

        @proxy_to("target", "missing", default=fallback)
        class Sample:
            target: object | None = None

        assert_type(Sample, type[Sample])

    def test_with_pre_processor(self) -> None:
        """GIVEN a pre-processor callable
        WHEN decorating a class
        THEN the class is still a type.
        """

        def processor(value: object) -> object:
            return value

        @proxy_to("target", "name", pre=processor)
        class Sample:
            target: object = None  # type: ignore[assignment][assignment]

        assert_type(Sample, type[Sample])

    def test_safe_false(self) -> None:
        """GIVEN safe=False
        WHEN decorating a class that already has the method
        THEN the class is still a type.
        """

        class Sample:
            target: object = None  # type: ignore[assignment][assignment]

            def existing(self) -> str:
                return "original"

        decorated = proxy_to("target", "existing", safe=False)(Sample)
        assert_type(decorated, type[Sample])

    def test_bind_none(self) -> None:
        """GIVEN bind=None (last positional arg is None)
        WHEN decorating a class
        THEN the class is still a type.
        """

        class Target:
            def method(self) -> str:
                return "hi"

        class Sample:
            target = Target()

        decorated = proxy_to("target", "method", None)(Sample)
        assert_type(decorated, type[Sample])

    def test_custom_getter(self) -> None:
        """GIVEN a custom getter
        WHEN decorating a class
        THEN the class is still a type.
        """

        def custom_getter(name: str) -> Callable[[Any], Any]:
            return lambda obj: getattr(obj, name)

        @proxy_to("target", "name", getter=custom_getter)
        class Sample:
            target: object = None  # type: ignore[assignment][assignment]

        assert_type(Sample, type[Sample])


# ------------------------------------------------------------------
# bound_property / BaseProperty — runtime type hints
# ------------------------------------------------------------------


class TestPrimitivesRuntimeHints:
    """GIVEN primitives classes
    WHEN inspecting annotations
    THEN constructors and methods are typed.
    """

    def test_bound_property_init_is_typed(self) -> None:
        hints = get_type_hints(bound_property.__init__)  # type: ignore[assignment][misc]
        assert "function" in hints

    def test_base_property_call_is_typed(self) -> None:
        hints = get_type_hints(BaseProperty.call)  # type: ignore[assignment][misc]
        assert "return" in hints
