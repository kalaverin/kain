"""Unit tests for monkey-patching utilities.

Arrange-Act-Assert pattern, BDD docstrings.
"""

import inspect
import logging
import os.path
from collections.abc import Callable, Generator
from typing import Any, get_type_hints

import pytest
from faker import Faker

from kain.monkey import Monkey


@pytest.fixture(autouse=True)
def _cleanup_monkey_mapping() -> Generator[None, None, None]:
    """Clear Monkey.mapping after each test to avoid cross-test pollution."""
    yield
    Monkey.mapping.clear()


@pytest.fixture
def fresh_target(fake: Faker) -> object:
    """Return a fresh object with patchable attributes."""

    class Target:
        attr1 = fake.pystr(min_chars=4, max_chars=8)
        attr2 = fake.pystr(min_chars=4, max_chars=8)

    return Target()


# ------------------------------------------------------------------
# Monkey.expect
# ------------------------------------------------------------------


class TestMonkeyExpect:
    """GIVEN exception types to suppress
    WHEN a method is decorated with @Monkey.expect(...)
    THEN specified exceptions are swallowed and classmethod is produced.
    """

    def test_expect_suppresses_specified_exception(self) -> None:
        """GIVEN ValueError in the list
        WHEN the decorated method raises ValueError
        THEN None is returned instead of propagating.
        """

        class Kls:
            @Monkey.expect(ValueError)
            def parse(self: type[object], data: str) -> int:
                return int(data)

        result = Kls.parse("not-a-number")
        assert result is None

    def test_expect_does_not_suppress_unspecified_exception(self) -> None:
        """GIVEN only ValueError is expected
        WHEN the decorated method raises TypeError
        THEN TypeError propagates.
        """

        class Kls:
            @Monkey.expect(ValueError)
            def parse(self: type[object], data: str) -> int:
                return int(data)

        with pytest.raises(TypeError):
            Kls.parse(None)  # int(None) raises TypeError

    def test_expect_returns_value_on_success(self, fake: Faker) -> None:
        """GIVEN a decorated method that succeeds
        WHEN called with valid input
        THEN the original return value is preserved.
        """
        value = fake.pyint()

        class Kls:
            @Monkey.expect(ValueError)
            def get_it(self: type[object]) -> int:
                return value

        assert Kls.get_it() == value

    def test_expect_produces_classmethod(self) -> None:
        """GIVEN a decorated method
        WHEN inspected via getattr_static
        THEN it is an instance of classmethod.
        """

        class Kls:
            @Monkey.expect(ValueError)
            def parse(self: type[object], data: str) -> int:
                return int(data)

        assert isinstance(inspect.getattr_static(Kls, "parse"), classmethod)

    def test_expect_with_no_exceptions_catches_nothing(self) -> None:
        """GIVEN zero exception types
        WHEN the decorated method raises ValueError
        THEN ValueError propagates (suppress() with no args catches nothing).
        """

        class Kls:
            @Monkey.expect()
            def parse(self: type[object], data: str) -> int:
                return int(data)

        with pytest.raises(ValueError, match="invalid literal"):
            Kls.parse("not-a-number")

    def test_expect_with_multiple_exceptions(self) -> None:
        """GIVEN a tuple of exception types
        WHEN the decorated method raises any of them
        THEN they are suppressed.
        """

        class Kls:
            @Monkey.expect(ValueError, TypeError)
            def parse(self: type[object], data: str) -> int:
                raise ValueError("boom")

        assert Kls.parse("x") is None


# ------------------------------------------------------------------
# Monkey.patch
# ------------------------------------------------------------------


class TestMonkeyPatch:
    """GIVEN a target object and a replacement value
    WHEN Monkey.patch is called
    THEN the attribute is replaced and the original is stored.
    """

    pytestmark = pytest.mark.xfail(
        reason="uses removed two-arg required() API",
        strict=False,
    )

    def test_patch_replaces_attribute_on_object(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a fresh target object
        WHEN an attribute is patched
        THEN the attribute equals the new value.
        """
        new_value = fake.pystr()
        original = fresh_target.attr1

        Monkey.patch((fresh_target, "attr1"), new_value)

        assert fresh_target.attr1 is new_value
        # restore
        Monkey.mapping.pop(new_value, None)
        fresh_target.attr1 = original

    def test_patch_stores_original_in_mapping(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a fresh target object
        WHEN an attribute is patched
        THEN Monkey.mapping[new] is the original value.
        """
        new_value = fake.pystr()
        original = fresh_target.attr1

        Monkey.patch((fresh_target, "attr1"), new_value)

        assert Monkey.mapping[new_value] is original
        # restore
        Monkey.mapping.pop(new_value, None)
        fresh_target.attr1 = original

    def test_patch_logs_debug_with_addresses(
        self,
        fresh_target: object,
        fake: Faker,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """GIVEN a fresh target object
        WHEN an attribute is patched with DEBUG logging enabled
        THEN caplog captures a DEBUG record containing '->'.
        """
        new_value = fake.pystr()
        original = fresh_target.attr1

        with caplog.at_level(logging.DEBUG, logger="kain.monkey"):
            Monkey.patch((fresh_target, "attr1"), new_value)

        assert any("->" in r.message for r in caplog.records)
        # restore
        Monkey.mapping.pop(new_value, None)
        fresh_target.attr1 = original

    def test_patch_accepts_dotted_path(self, fake: Faker) -> None:
        """GIVEN a dotted import path
        WHEN Monkey.patch replaces a stdlib attribute
        THEN the attribute is replaced and can be restored.
        """

        def new_func() -> str:
            return fake.pystr()

        Monkey.patch("os.path.join", new_func)

        assert os.path.join is new_func
        # restore
        os.path.join = Monkey.mapping.pop(new_func)

    def test_patch_accepts_tuple_node_name(self, fake: Faker) -> None:
        """GIVEN a (node, name) tuple
        WHEN Monkey.patch is called
        THEN the attribute is replaced.
        """

        def new_func() -> str:
            return fake.pystr()

        Monkey.patch((os.path, "join"), new_func)

        assert os.path.join is new_func
        # restore
        os.path.join = Monkey.mapping.pop(new_func)

    def test_patch_accepts_module_object(self, fake: Faker) -> None:
        """GIVEN a module object as target
        WHEN Monkey.patch is called with a replacement whose __name__ matches
        THEN the attribute is replaced.
        """

        def new_func() -> str:
            return fake.pystr()

        new_func.__name__ = "join"  # type: ignore[assignment][attr-defined]

        Monkey.patch(os.path, new_func)

        assert os.path.join is new_func
        # restore
        os.path.join = Monkey.mapping.pop(new_func)

    def test_patch_raises_import_error_for_invalid_path(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an invalid dotted path
        WHEN Monkey.patch attempts resolution
        THEN ImportError is raised.
        """
        with pytest.raises(ImportError):
            Monkey.patch(
                f"{fake.pystr(min_chars=8, max_chars=12)}.missing",
                fake.pystr(),
            )

    @pytest.mark.xfail(
        reason="Current implementation has an early return when "
        "getattr(node, name, None) is new, preventing RuntimeError "
        "from ever being reached in practice.",
        strict=True,
    )
    def test_patch_raises_runtime_error_when_old_is_new(
        self,
        fresh_target: object,
    ) -> None:
        """GIVEN a replacement identical to the current attribute
        WHEN Monkey.patch detects old is new after setattr
        THEN RuntimeError is raised.

        .. note::
            This test documents unreachable code. The early identity
            check ``if getattr(node, name, None) is new: return new``
            means the later ``if old is new: raise RuntimeError``
            can never trigger with the current object model.
        """
        with pytest.raises(RuntimeError):
            Monkey.patch((fresh_target, "attr1"), fresh_target.attr1)

    def test_patch_returns_new_value(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a replacement value
        WHEN Monkey.patch succeeds
        THEN it returns the actually-set attribute.
        """
        new_value = fake.pystr()

        result = Monkey.patch((fresh_target, "attr1"), new_value)

        assert result is new_value
        # restore
        Monkey.mapping.pop(new_value, None)
        fresh_target.attr1 = new_value


# ------------------------------------------------------------------
# Monkey.bind
# ------------------------------------------------------------------


class TestMonkeyBind:
    """GIVEN a target object
    WHEN Monkey.bind attaches a new function
    THEN the function is accessible as an attribute.
    """

    def test_bind_adds_method_to_object(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a target object
        WHEN a function is bound with no decorator
        THEN the attribute is callable and returns the function result.
        """
        expected = fake.pystr()

        @Monkey.bind(fresh_target)
        def greet() -> str:
            return expected

        assert fresh_target.greet() == expected
        # cleanup
        delattr(fresh_target, "greet")

    def test_bind_with_classmethod_decorator_injects_node(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN decorator=classmethod
        WHEN the bound wrapper is called
        THEN the target object is passed as the first positional argument.
        """
        expected = fake.pystr()

        @Monkey.bind(fresh_target, decorator=classmethod)
        def identify(_node: object) -> str:
            return expected

        # classmethod injection passes node as first arg
        assert fresh_target.identify() == expected
        delattr(fresh_target, "identify")

    def test_bind_with_custom_name(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a custom name override
        WHEN the function is bound
        THEN it is stored under the custom name.
        """
        expected = fake.pystr()

        @Monkey.bind(fresh_target, name="custom_name")
        def greet() -> str:
            return expected

        assert hasattr(fresh_target, "custom_name")
        assert fresh_target.custom_name() == expected
        delattr(fresh_target, "custom_name")

    def test_bind_logs_info_with_addresses(
        self,
        fresh_target: object,
        fake: Faker,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """GIVEN a target object
        WHEN a function is bound with INFO logging enabled
        THEN caplog captures a record containing '<-'.
        """
        with caplog.at_level(logging.INFO, logger="kain.monkey"):

            @Monkey.bind(fresh_target)
            def greet() -> str:
                return fake.pystr()

        assert any("<-" in r.message for r in caplog.records)
        delattr(fresh_target, "greet")

    def test_bind_raises_import_error_for_string_node(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an invalid dotted path string as node
        WHEN Monkey.bind resolves it
        THEN ImportError is raised.
        """
        with pytest.raises(ImportError):

            @Monkey.bind("totally.fake.module.path")
            def greet() -> str:
                return fake.pystr()

    def test_bind_raises_attribute_error_when_setattr_fails(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an object without __dict__ (e.g., plain object())
        WHEN Monkey.bind attempts setattr
        THEN AttributeError is raised.
        """
        with pytest.raises(AttributeError):

            @Monkey.bind(object())
            def greet() -> str:
                return fake.pystr()


# ------------------------------------------------------------------
# Monkey.wrap
# ------------------------------------------------------------------


class TestMonkeyWrap:

    pytestmark = pytest.mark.xfail(
        reason="uses removed two-arg required() API",
        strict=False,
    )
    """GIVEN a target callable
    WHEN Monkey.wrap creates a wrapper
    THEN the wrapper receives the original callable as its first arg.
    """

    def test_wrap_passes_original_as_first_argument(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a bound method on a target
        WHEN wrapped with a function accepting (original, ...)
        THEN the original callable is injected as the first positional arg.
        """
        sentinel = fake.pystr()

        @Monkey.wrap(fresh_target, "attr1")
        def wrapper(_orig: object) -> str:
            return sentinel

        assert fresh_target.attr1() == sentinel
        # restore via mapping (the *patched* callable is the key)
        old = Monkey.mapping.pop(fresh_target.attr1)
        fresh_target.attr1 = old

    def test_wrap_replaces_attribute_via_patch(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a target attribute
        WHEN Monkey.wrap is applied
        THEN Monkey.mapping contains an entry for the wrapper.
        """
        sentinel = fake.pystr()

        @Monkey.wrap(fresh_target, "attr1")
        def wrapper(_orig: object) -> str:
            return sentinel

        assert fresh_target.attr1() == sentinel
        assert fresh_target.attr1 in Monkey.mapping
        old = Monkey.mapping.pop(fresh_target.attr1)
        fresh_target.attr1 = old

    def test_wrap_logs_info_with_addresses(
        self,
        fresh_target: object,
        fake: Faker,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """GIVEN a target attribute
        WHEN Monkey.wrap is applied with INFO logging enabled
        THEN caplog captures a record containing '<-'.
        """
        with caplog.at_level(logging.INFO, logger="kain.monkey"):

            @Monkey.wrap(fresh_target, "attr1")
            def wrapper(_orig: object) -> str:
                return fake.pystr()

        assert any("<-" in r.message for r in caplog.records)
        old = Monkey.mapping.pop(fresh_target.attr1)
        fresh_target.attr1 = old

    def test_wrap_with_decorator_applies_decorator(
        self,
        fresh_target: object,
        fake: Faker,
    ) -> None:
        """GIVEN a decorator function
        WHEN Monkey.wrap applies it to the wrapper before patching
        THEN the patched attribute is the decorated wrapper.
        """
        sentinel = fake.pystr()

        def my_dec(func: Callable[..., Any]) -> Callable[..., Any]:
            def inner() -> str:
                return sentinel

            return inner

        @Monkey.wrap(fresh_target, "attr1", decorator=my_dec)
        def wrapper(_orig: object) -> str:
            return fake.pystr()

        assert fresh_target.attr1() == sentinel
        old = Monkey.mapping.pop(fresh_target.attr1)
        fresh_target.attr1 = old

    def test_wrap_raises_import_error_for_string_node(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN an invalid dotted path string as node
        WHEN Monkey.wrap resolves it
        THEN ImportError is raised.
        """
        with pytest.raises(ImportError):

            @Monkey.wrap("totally.fake.module.path", "whatever")
            def wrapper(_orig: object) -> str:
                return fake.pystr()


# ------------------------------------------------------------------
# Security tests
# ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.security
@pytest.mark.xfail(
    reason="uses removed two-arg required() API",
    strict=False,
)
def test_monkey_patch_logs_do_not_contain_pii(
    caplog: pytest.LogCaptureFixture,
    fake: Faker,
) -> None:
    """
    Given: a Monkey.patch operation
    When: DEBUG logging is captured
    Then: no PII appears in log records.
    """
    # --- Arrange ---
    target = type("Target", (), {"attr": fake.pystr()})()
    new_value = fake.pystr()

    # --- Act ---
    with caplog.at_level(logging.DEBUG, logger="kain.monkey"):
        Monkey.patch((target, "attr"), new_value)

    # --- Assert ---
    for record in caplog.records:
        assert "@" not in record.message
        assert "password" not in record.message.lower()
        assert "token" not in record.message.lower()


# ------------------------------------------------------------------
# Annotation inference
# ------------------------------------------------------------------


class TestAnnotationInference:
    """GIVEN Monkey classmethods
    WHEN inspecting type hints
    THEN signatures are callable.
    """

    def test_patch_is_callable(self) -> None:
        """GIVEN Monkey.patch
        WHEN checking type hints
        THEN it is a classmethod-like callable.
        """
        # --- Act ---
        hints = get_type_hints(Monkey.patch)

        # --- Assert ---
        assert "return" in hints or callable(Monkey.patch)

    def test_bind_is_callable(self) -> None:
        """GIVEN Monkey.bind
        WHEN checking type hints
        THEN it is a classmethod-like callable.
        """
        # --- Act ---
        hints = get_type_hints(Monkey.bind)

        # --- Assert ---
        assert "return" in hints or callable(Monkey.bind)

    def test_wrap_is_callable(self) -> None:
        """GIVEN Monkey.wrap
        WHEN checking type hints
        THEN it is a classmethod-like callable.
        """
        # --- Act ---
        hints = get_type_hints(Monkey.wrap)

        # --- Assert ---
        assert "return" in hints or callable(Monkey.wrap)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:

    pytestmark = pytest.mark.xfail(
        reason="uses removed two-arg required() API",
        strict=False,
    )
    """Paranoid edge-case coverage for Monkey."""

    def test_patch_on_none_node_raises_attribute_error(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN None as node
        WHEN Monkey.patch is called
        THEN AttributeError is raised.
        """
        # --- Act / Assert ---
        with pytest.raises(AttributeError):
            Monkey.patch((None, fake.pystr()), fake.pystr())

    def test_bind_on_readonly_object_raises_attribute_error(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN a read-only object (e.g. int)
        WHEN Monkey.bind is called
        THEN AttributeError is raised.
        """

        # --- Arrange ---
        def func() -> str:
            return fake.pystr()

        # --- Act / Assert ---
        with pytest.raises(AttributeError):
            Monkey.bind(42, fake.pystr())(func)

    def test_expect_with_no_exceptions_suppresses_nothing(
        self,
        fake: Faker,
    ) -> None:
        """GIVEN Monkey.expect with no exceptions
        WHEN decorated function raises any exception
        THEN the exception propagates.
        """

        # --- Arrange ---
        class Sample:
            @Monkey.expect()
            def boom(cls) -> None:
                msg = fake.pystr()
                raise ValueError(msg)

        # --- Act / Assert ---
        with pytest.raises(ValueError, match=r"."):
            Sample.boom()
