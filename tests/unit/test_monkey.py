"""Unit tests for the Monkey patching namespace."""

from __future__ import annotations

import logging
import os.path
from collections.abc import Callable, Generator
from contextlib import contextmanager
from types import ModuleType

import pytest
from faker import Faker

from kain.importer import required
from kain.monkey import Monkey

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _reset_namespace_mappings() -> Generator[None]:
    """Save and restore Monkey.mapping around each test."""
    saved = dict(Monkey.mapping)
    Monkey.mapping.clear()
    yield
    Monkey.mapping.clear()
    Monkey.mapping.update(saved)


@pytest.fixture
def fresh_callable_target(
    monkeypatch_target_factory: Callable[..., object],
) -> object:
    """Fresh object with a callable ``func`` attribute."""
    target = monkeypatch_target_factory()
    target.func = lambda: "original"
    return target


@pytest.fixture
def fresh_class_target() -> type:
    """Fresh class with a classmethod ``method``."""

    class Node:
        @classmethod
        def method(cls) -> str:
            return "original"

    return Node


def _make_join_replacement(fake: Faker) -> Callable[..., str]:
    """Return a function named ``join`` for os.path patching."""

    def replacement(*args: object, **kwargs: object) -> str:
        return fake.pystr()

    replacement.__name__ = "join"
    replacement.__qualname__ = "join"
    return replacement


def _patch_target(
    kind: str,
    fresh_target: object,
) -> tuple[object, object, str, object]:
    """Decode a patch matrix kind into (target, node, name, original)."""
    if kind == "tuple_attr1":
        return (
            (fresh_target, "attr1"),
            fresh_target,
            "attr1",
            fresh_target.attr1,
        )
    if kind == "tuple_missing_attr":
        return (
            (fresh_target, "missing_attr"),
            fresh_target,
            "missing_attr",
            None,
        )
    if kind == "string_os.path.join":
        return "os.path.join", os.path, "join", os.path.join
    if kind == "module_os.path":
        return os.path, os.path, "join", os.path.join
    raise ValueError(kind)


def _wrap_target(
    kind: str,
    fresh_target: object,
    fresh_class_target: type,
) -> tuple[object, str, object]:
    """Decode a wrap matrix kind into (node, name, original)."""
    if kind == "object_func":
        return fresh_target, "func", fresh_target.func
    if kind == "class_method":
        return fresh_class_target, "method", fresh_class_target.method
    if kind == "string_os.path_join":
        return "os.path", "join", os.path.join
    raise ValueError(kind)


@contextmanager
def _apply_patch(
    cls: type[Monkey],
    target: str | ModuleType | tuple[object, str],
    new: object,
) -> Generator[object]:
    """Call ``cls.replace`` and restore the attribute afterwards."""
    result = cls.replace(target, new)
    try:
        yield result
    finally:
        original = cls.mapping.pop(result, None)
        if original is not None:
            restored = cls.replace(target, original)
            cls.mapping.pop(restored, None)


@contextmanager
def _apply_wrap(
    cls: type[Monkey],
    node: str | object,
    name: str,
    decorator: Callable[..., object] | None,
    wrapper_func: Callable[..., object],
) -> Generator[tuple[object, Callable[..., object]]]:
    """Call ``cls.wrap`` and restore the attribute afterwards."""
    if isinstance(node, str):
        node = required(node)
    original = getattr(node, name)
    wrapper = cls.wrap(node, name=name, decorator=decorator)(wrapper_func)
    set_value = getattr(node, name)
    try:
        yield set_value, wrapper
    finally:
        stored = cls.mapping.pop(set_value, None)
        setattr(node, name, stored if stored is not None else original)


@contextmanager
def _apply_bind(
    cls: type[Monkey],
    node: str | object,
    name: str | None,
    decorator: Callable[..., object] | None,
    func: Callable[..., object],
) -> Generator[Callable[..., object]]:
    """Call ``cls.bind`` and remove the attribute afterwards."""
    bound_name = name or func.__name__
    original = (
        getattr(node, bound_name, None) if hasattr(node, bound_name) else None
    )
    wrapper = cls.bind(node, name=name, decorator=decorator)(func)
    try:
        yield wrapper
    finally:
        if hasattr(node, bound_name):
            delattr(node, bound_name)
        if original is not None:
            setattr(node, bound_name, original)


@pytest.mark.parametrize(
    "kind, new_kind",
    (
        pytest.param("tuple_attr1", "replace", id="tuple-replace"),
        pytest.param("string_os.path.join", "replace", id="string-replace"),
        pytest.param("module_os.path", "replace", id="module-replace"),
        pytest.param("tuple_attr1", "same", id="tuple-short"),
        pytest.param("string_os.path.join", "same", id="string-short"),
        pytest.param("module_os.path", "same", id="module-short"),
    ),
)
def test_patch_replacement_matrix(
    kind: str,
    new_kind: str,
    monkey_target: object,
    fake: Faker,
) -> None:
    """GIVEN a target form
    WHEN Monkey.patch replaces the attribute
    THEN the new value is set and the original is stored.
    """
    target, node, name, original = _patch_target(kind, monkey_target)
    if new_kind == "replace":
        new_value: object = (
            _make_join_replacement(fake) if name == "join" else fake.pystr()
        )
    else:
        # Identity short-circuit: patch with the current value itself.
        new_value = original

    with _apply_patch(Monkey, target, new_value) as result:
        assert result is new_value
        assert getattr(node, name) is new_value
        if new_kind == "replace":
            assert Monkey.mapping[result] is original
        else:
            assert new_value not in Monkey.mapping


@pytest.mark.parametrize(
    "kind, expected_exc",
    (
        pytest.param("string_invalid", ImportError, id="invalid-string"),
        pytest.param("tuple_missing_attr", AttributeError, id="missing-attr"),
        pytest.param("module_no_name", AttributeError, id="module-no-name"),
    ),
)
def test_patch_error_matrix(
    kind: str,
    expected_exc: type[BaseException],
    monkey_target: object,
    fake: Faker,
) -> None:
    """GIVEN an invalid target
    WHEN Monkey.patch is called
    THEN the expected exception is raised.
    """
    if kind == "string_invalid":
        target: object = f"{fake.pystr(min_chars=8, max_chars=12)}.missing"
    elif kind == "tuple_missing_attr":
        target = (monkey_target, "missing_attr")
    elif kind == "module_no_name":
        target = os.path
    else:
        raise ValueError(kind)

    new: object = fake.pystr() if kind != "module_no_name" else object()
    with pytest.raises(expected_exc):
        Monkey.replace(target, new)


@pytest.mark.parametrize(
    "decorator_kind, name_kind",
    (
        pytest.param(None, None, id="bind-plain"),
        pytest.param("classmethod", None, id="bind-classmethod"),
        pytest.param(None, "custom", id="bind-custom-name"),
        pytest.param("custom", None, id="bind-decorator"),
    ),
)
def test_bind_attachment_matrix(
    decorator_kind: str | None,
    name_kind: str | None,
    monkey_target: object,
    fake: Faker,
) -> None:
    """GIVEN a target object
    WHEN a function is bound with optional name/decorator
    THEN it is accessible and cleanup restores state.
    """
    expected = fake.pystr()
    name = "custom_name" if name_kind == "custom" else None
    decorated: list[bool] = []

    decorator: Callable[..., object] | None = None
    if decorator_kind == "classmethod":
        decorator = classmethod
    elif decorator_kind == "custom":

        def custom_decorator(
            fn: Callable[..., object],
        ) -> Callable[..., object]:
            decorated.append(True)
            return fn

        decorator = custom_decorator

    if decorator_kind == "classmethod":

        def func(node: object) -> str:
            return expected

    else:

        def func() -> str:
            return expected

    with _apply_bind(Monkey, monkey_target, name, decorator, func):
        bound_name = name or func.__name__
        assert hasattr(monkey_target, bound_name)
        if decorator_kind == "custom":
            # Bind does not apply arbitrary decorators; only classmethod
            # injection is handled.
            assert decorated == []
        assert getattr(monkey_target, bound_name)() == expected


@pytest.mark.parametrize(
    "kind, decorator_kind",
    (
        pytest.param("object_func", None, id="wrap-object"),
        pytest.param("class_method", None, id="wrap-class"),
        pytest.param("string_os.path_join", None, id="wrap-string"),
        pytest.param("object_func", "custom", id="wrap-decorator"),
    ),
)
def test_wrap_replacement_matrix(
    kind: str,
    decorator_kind: str | None,
    fresh_callable_target: object,
    fresh_class_target: type,
) -> None:
    """GIVEN a callable attribute
    WHEN it is wrapped
    THEN the wrapper receives the original and the attribute is restored.
    """
    node, name, _original = _wrap_target(
        kind,
        fresh_callable_target,
        fresh_class_target,
    )
    sentinel = "wrap-sentinel"

    decorator: Callable[..., object] | None = None
    if decorator_kind == "custom":

        def custom_decorator(
            fn: Callable[..., object],
        ) -> Callable[..., object]:
            def inner(*args: object, **kwargs: object) -> str:
                return f"decorated:{fn(*args, **kwargs)}"

            return inner

        decorator = custom_decorator

    def wrapper(wrapped: object, *args: object, **kwargs: object) -> str:
        result = wrapped(*args, **kwargs) if name == "join" else wrapped()
        return f"{sentinel}:{result}"

    with _apply_wrap(Monkey, node, name, decorator, wrapper) as (
        set_value,
        _wrapped,
    ):
        if decorator_kind == "custom":
            assert set_value is not _wrapped
        assert set_value in Monkey.mapping
        if name == "method":
            obj = fresh_class_target()
            assert obj.method() == f"{sentinel}:original"
        elif name == "join":
            assert os.path.join("a") == f"{sentinel}:a"  # noqa: PTH118
        elif decorator_kind == "custom":
            assert node.func() == f"decorated:{sentinel}:original"
        else:
            assert node.func() == f"{sentinel}:original"


@pytest.mark.parametrize(
    "node",
    (pytest.param("totally.fake.module.path", id="wrap-invalid"),),
)
def test_wrap_error_matrix(
    node: str,
    fake: Faker,
) -> None:
    """GIVEN an invalid dotted path
    WHEN Monkey.wrap resolves it
    THEN ImportError is raised.
    """

    def wrapper(wrapped: object) -> str:
        return fake.pystr()

    with pytest.raises(ImportError):
        Monkey.wrap(node, name="whatever")(wrapper)


@pytest.mark.parametrize(
    "kind",
    (pytest.param("tuple_attr1", id="patch-log"),),
)
def test_patch_logs_matrix(
    kind: str,
    monkey_target: object,
    fake: Faker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GIVEN debug logging enabled
    WHEN Monkey.patch runs
    THEN a structured 'attribute patched' record is emitted.
    """
    target, _node, name, _original = _patch_target(kind, monkey_target)
    new_value: object = (
        _make_join_replacement(fake) if name == "join" else fake.pystr()
    )
    with (
        caplog.at_level(logging.DEBUG, logger=Monkey.__module__),
        _apply_patch(Monkey, target, new_value),
    ):
        pass

    assert any(
        record.message == "attribute replaced"
        and getattr(record, "before", "") != ""
        and getattr(record, "after", "") != ""
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "cls",
    (Monkey,),
    ids=("monkey",),
)
def test_bind_logs_matrix(
    cls: type[Monkey],
    monkey_target: object,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GIVEN debug logging enabled
    WHEN Monkey.bind runs
    THEN a structured 'attribute replaced with' record is emitted.
    """

    def func() -> str:
        return "bound"

    with (
        caplog.at_level(logging.DEBUG, logger=cls.__module__),
        _apply_bind(cls, monkey_target, None, None, func),
    ):
        pass

    assert any(
        record.message == "attribute bounded with"
        and getattr(record, "before", "") != ""
        and getattr(record, "after", "") != ""
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "kind",
    (pytest.param("object_func", id="wrap-log"),),
)
def test_wrap_logs_matrix(
    kind: str,
    fresh_callable_target: object,
    fresh_class_target: type,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GIVEN debug logging enabled
    WHEN Monkey.wrap runs
    THEN a structured 'attribute replaced with' record is emitted.
    """
    node, name, _original = _wrap_target(
        kind,
        fresh_callable_target,
        fresh_class_target,
    )

    def wrapper(wrapped: object) -> str:
        return "wrapped"

    with (
        caplog.at_level(logging.DEBUG, logger=Monkey.__module__),
        _apply_wrap(Monkey, node, name, None, wrapper),
    ):
        pass

    assert any(
        record.message == "attribute wrapped with"
        and getattr(record, "before", "") != ""
        and getattr(record, "after", "") != ""
        for record in caplog.records
    )


@pytest.mark.parametrize(
    "cls",
    (Monkey,),
    ids=("monkey",),
)
def test_wrap_staticmethod_matrix(
    cls: type[Monkey],
    fake: Faker,
) -> None:
    """GIVEN a staticmethod
    WHEN it is wrapped
    THEN the wrapper receives the original callable.
    """
    sentinel = fake.pystr()

    class Node:
        @staticmethod
        def static() -> str:
            return "original"

    def wrapper(wrapped: object) -> str:
        return f"{sentinel}:{wrapped()}"

    with _apply_wrap(cls, Node, "static", None, wrapper):
        assert Node.static() == f"{sentinel}:original"


@pytest.mark.parametrize(
    "kind, expected_exc",
    (
        pytest.param("bind_object", AttributeError, id="bind-object"),
        pytest.param("bind_int", AttributeError, id="bind-int"),
        pytest.param("patch_object", AttributeError, id="patch-object"),
        pytest.param("patch_built_in", TypeError, id="patch-built-in"),
        pytest.param("patch_none", AttributeError, id="patch-none"),
    ),
)
def test_patch_and_bind_error_surfaces(
    kind: str,
    expected_exc: type[BaseException],
    fake: Faker,
) -> None:
    """GIVEN an unwritable or invalid target
    WHEN Monkey.patch / Monkey.bind is called
    THEN the expected exception is raised and no state leaks.
    """
    if kind == "bind_object":
        target = Monkey.bind(object(), fake.pystr())
    elif kind == "bind_int":
        target = Monkey.bind(42, fake.pystr())
    elif kind == "patch_object":
        target = (object(), fake.pystr())
    elif kind == "patch_built_in":
        target = (str, "join")
    elif kind == "patch_none":
        target = (None, fake.pystr())
    else:
        raise ValueError(kind)

    if kind.startswith("bind_"):
        with pytest.raises(expected_exc):
            target(lambda: None)
    else:
        with pytest.raises(expected_exc):
            Monkey.replace(target, fake.pystr())

    assert not Monkey.mapping


@pytest.mark.parametrize(
    "cls",
    (Monkey,),
    ids=("monkey",),
)
def test_wrap_property_and_descriptors(
    cls: type[Monkey],
    fake: Faker,
) -> None:
    """GIVEN a property descriptor
    WHEN it is wrapped
    THEN the wrapper receives the property and the instance.
    """
    sentinel = fake.pystr()

    class Node:
        @property
        def value(self) -> str:
            return "original"

    def wrapper(wrapped: object, self: object) -> str:
        return f"{sentinel}:{wrapped.__get__(self, type(self))}"

    with _apply_wrap(cls, Node, "value", None, wrapper):
        obj = Node()
        assert obj.value() == f"{sentinel}:original"


@pytest.mark.parametrize(
    "kind",
    (
        pytest.param("patch_invalid_string", id="patch-invalid-string"),
        pytest.param("patch_missing_attr", id="patch-missing-attr"),
        pytest.param("wrap_missing_attr", id="wrap-missing-attr"),
    ),
)
def test_partial_failure_does_not_leak_state(
    kind: str,
    monkey_target: object,
    fake: Faker,
) -> None:
    """GIVEN an operation that fails before mutating mapping
    WHEN the exception is raised
    THEN Monkey.mapping stays empty.
    """
    if kind == "patch_invalid_string":
        target: object = f"{fake.pystr(min_chars=8, max_chars=12)}.missing"
    elif kind in {"patch_missing_attr", "wrap_missing_attr"}:
        target = (monkey_target, "missing_attr")
    else:
        raise ValueError(kind)

    def wrapper(wrapped: object) -> str:
        return fake.pystr()

    if kind == "wrap_missing_attr":

        def run() -> None:
            Monkey.wrap(target[0], name=target[1])(wrapper)
    else:

        def run() -> None:
            Monkey.replace(target, fake.pystr())

    with pytest.raises((ImportError, AttributeError)):
        run()

    assert not Monkey.mapping


@pytest.mark.parametrize(
    "cls",
    (Monkey,),
    ids=("monkey",),
)
def test_logs_do_not_contain_pii(
    cls: type[Monkey],
    fake: Faker,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """GIVEN a patch operation with PII-like data
    WHEN logs are captured
    THEN no PII appears in log records.
    """
    target = type("Target", (), {"attr": fake.pystr()})()
    new_value = f"{fake.email()} {fake.password()}"

    with (
        caplog.at_level(logging.DEBUG, logger=cls.__module__),
        _apply_patch(cls, (target, "attr"), new_value),
    ):
        pass

    for record in caplog.records:
        assert "@" not in record.message
        assert "password" not in record.message.lower()
        assert "token" not in record.message.lower()
        assert getattr(record, "before", "") != new_value
        assert getattr(record, "after", "") != new_value
