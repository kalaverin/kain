from __future__ import annotations

import logging
import os
import sys
import types
from contextlib import suppress
from pathlib import Path
from types import ModuleType
from typing import get_type_hints

import pytest
from faker import Faker

from kain.importer import (
    add_path,
    cached_import,
    get_child,
    get_module,
    get_path,
    import_object,
    optional,
    required,
)


@pytest.mark.unit
def test_get_module_imports_full_path() -> None:
    """
    Given: a dotted path to an attribute inside a module
    When: calling get_module
    Then: returns the module and remaining attribute components
    """
    # --- Act ---
    module, remaining = get_module("os.path.join")

    # --- Assert ---
    assert isinstance(module, ModuleType)
    assert remaining == ("join",)


@pytest.mark.unit
def test_get_module_imports_module_only() -> None:
    """
    Given: a dotted path to a module
    When: calling get_module
    Then: returns the module and empty remaining components
    """
    # --- Act ---
    module, remaining = get_module("os.path")

    # --- Assert ---
    assert isinstance(module, ModuleType)
    assert remaining == ()


@pytest.mark.unit
def test_get_module_raises_import_error_for_missing(fake: Faker) -> None:
    """
    Given: an invalid module path
    When: calling get_module
    Then: raises ImportError with descriptive message
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act / Assert ---
    with pytest.raises(ImportError, match="does not exist"):
        get_module(invalid_path)


@pytest.mark.unit
def test_get_child_gets_attribute() -> None:
    """
    Given: a parent module and a child attribute name
    When: calling get_child
    Then: returns the attribute
    """
    # --- Act ---
    result = get_child("os.path.join", os.path, "join")

    # --- Assert ---
    assert result is os.path.join


@pytest.mark.unit
def test_get_child_raises_import_error_for_missing_attr(fake: Faker) -> None:
    """
    Given: a parent object without the requested attribute
    When: calling get_child
    Then: raises ImportError
    """
    missing_name = fake.pystr(min_chars=8, max_chars=12)

    # --- Act / Assert ---
    with pytest.raises(ImportError, match="hasn't member"):
        get_child("os.path.missing", os.path, missing_name)


@pytest.mark.unit
def test_get_child_detects_circular_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Given: a partially initialized module with no public attrs
    When: calling get_child for a missing attribute
    Then: raises ImportError mentioning circular import
    """
    mod = types.ModuleType("fake_mod")
    for attr in (
        "__builtins__",
        "__cached__",
        "__doc__",
        "__file__",
        "__loader__",
        "__package__",
        "__path__",
        "__spec__",
    ):
        setattr(mod, attr, None)

    monkeypatch.setitem(sys.modules, "fake_mod", mod)

    # --- Act / Assert ---
    with pytest.raises(ImportError, match="circular import"):
        get_child("fake_mod.missing", mod, "missing")


@pytest.mark.unit
def test_import_object_imports_module() -> None:
    """
    Given: a dotted module path
    When: calling import_object
    Then: returns the module
    """
    # --- Act ---
    result = import_object("os.path")

    # --- Assert ---
    assert isinstance(result, ModuleType)


@pytest.mark.unit
def test_import_object_imports_attribute() -> None:
    """
    Given: a dotted attribute path
    When: calling import_object
    Then: returns the attribute
    """
    # --- Act ---
    result = import_object("os.path.join")

    assert result is os.path.join


@pytest.mark.unit
def test_import_object_with_parent_object() -> None:
    """
    Given: a parent object and an attribute path
    When: calling import_object with both args
    Then: returns the resolved attribute
    """
    # --- Act ---
    result = import_object("join", os.path)

    # --- Assert ---
    assert result is os.path.join


@pytest.mark.unit
def test_import_object_with_bytes_path() -> None:
    """
    Given: a bytes path
    When: calling import_object
    Then: coerces bytes to str and imports successfully
    """
    # --- Act ---
    result = import_object(b"os.path")

    # --- Assert ---
    assert isinstance(result, ModuleType)


@pytest.mark.unit
def test_import_object_raises_type_error_when_both_none() -> None:
    """
    Given: both arguments are None
    When: calling import_object
    Then: raises TypeError
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match="all arguments are None"):
        import_object(None, None)


@pytest.mark.unit
def test_import_object_raises_type_error_path_not_str_something_none() -> None:
    """
    Given: a non-string path and None second argument
    When: calling import_object
    Then: raises TypeError
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match="isn't str"):
        import_object(123, None)


@pytest.mark.unit
def test_import_object_type_error_contains_who_is_info() -> None:
    """
    Given: a non-string path and None second argument
    When: calling import_object
    Then: the TypeError message includes Who.Is(path) type info.
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match="int"):
        import_object(123, None)


@pytest.mark.unit
def test_import_object_raises_import_error_for_missing(fake: Faker) -> None:
    """
    Given: an invalid import path
    When: calling import_object
    Then: raises ImportError
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act / Assert ---
    with pytest.raises(ImportError):
        import_object(invalid_path)


@pytest.mark.unit
def test_cached_import_returns_same_object_on_second_call() -> None:
    """
    Given: a valid import path
    When: calling cached_import twice with the same args
    Then: returns the identical object
    """
    # --- Act ---
    first = cached_import("os.path.join")
    second = cached_import("os.path.join")

    # --- Assert ---
    assert first is second


@pytest.mark.unit
def test_cached_import_different_args_return_different_objects() -> None:
    """
    Given: two different valid import paths
    When: calling cached_import for each
    Then: returns different objects
    """
    # --- Act ---
    first = cached_import("os.path.join")
    second = cached_import("os.path.exists")

    # --- Assert ---
    assert first is not second


@pytest.mark.unit
def test_required_imports_existing_module() -> None:
    """
    Given: an existing import path
    When: calling required
    Then: returns the imported object
    """
    # --- Act ---
    result = required("os.path.join")

    assert result is os.path.join


@pytest.mark.unit
def test_required_raises_import_error_for_missing(fake: Faker) -> None:
    """
    Given: an invalid import path
    When: calling required
    Then: raises ImportError with helpful message
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act / Assert ---
    with pytest.raises(ImportError):
        required(invalid_path)


@pytest.mark.unit
def test_required_logs_warning_on_failure_when_quiet_false(
    caplog: pytest.LogCaptureFixture,
    fake: Faker,
) -> None:
    """
    Given: an invalid import path with quiet=False
    When: calling required
    Then: logs a warning
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act ---
    with (
        caplog.at_level(logging.WARNING, logger="kain.importer"),
        suppress(ImportError),
    ):
        required(invalid_path, quiet=False)

    # --- Assert ---
    assert any("couldn't import required" in r.message for r in caplog.records)


@pytest.mark.unit
def test_required_does_not_log_when_quiet_true(
    caplog: pytest.LogCaptureFixture,
    fake: Faker,
) -> None:
    """
    Given: an invalid import path with quiet=True
    When: calling required
    Then: no warning is logged
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act ---
    with (
        caplog.at_level(logging.WARNING, logger="kain.importer"),
        suppress(ImportError),
    ):
        required(invalid_path, quiet=True)

    # --- Assert ---
    assert not any(
        "couldn't import required" in r.message for r in caplog.records
    )


@pytest.mark.unit
def test_required_returns_default_when_throw_false(fake: Faker) -> None:
    """
    Given: an invalid import path with throw=False
    When: calling required
    Then: returns the default value
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"
    default_value = fake.pystr()

    # --- Act ---
    result = required(invalid_path, throw=False, default=default_value)

    # --- Assert ---
    assert result == default_value


@pytest.mark.unit
def test_required_suggests_package_when_base_not_in_sys_modules(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """
    Given: a missing module whose base name is not in sys.modules
    When: calling required
    Then: the error message suggests a PyPI package name
    """
    # --- Act / Assert ---
    with (
        caplog.at_level(logging.WARNING, logger="kain.importer"),
        pytest.raises(ImportError, match="need extra package"),
    ):
        required("magic.parser", quiet=False)


@pytest.mark.unit
def test_optional_returns_none_on_failure(fake: Faker) -> None:
    """
    Given: an invalid import path
    When: calling optional
    Then: returns None
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act ---
    result = optional(invalid_path)

    # --- Assert ---
    assert result is None


@pytest.mark.unit
def test_optional_returns_default_on_failure(fake: Faker) -> None:
    """
    Given: an invalid import path with a default
    When: calling optional
    Then: returns the default
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"
    default_value = fake.pystr()

    # --- Act ---
    result = optional(invalid_path, default=default_value)

    # --- Assert ---
    assert result == default_value


@pytest.mark.unit
def test_optional_does_not_log_by_default(
    caplog: pytest.LogCaptureFixture,
    fake: Faker,
) -> None:
    """
    Given: an invalid import path
    When: calling optional
    Then: no warning is logged
    """
    # --- Arrange ---
    invalid_path = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act ---
    with caplog.at_level(logging.WARNING, logger="kain.importer"):
        optional(invalid_path)

    # --- Assert ---
    assert not any(
        "couldn't import required" in r.message for r in caplog.records
    )


@pytest.mark.unit
def test_get_path_dot_returns_current_directory() -> None:
    """
    Given: a dot path
    When: calling get_path
    Then: returns Path(".")
    """
    # --- Act ---
    result = get_path(".")

    # --- Assert ---
    assert result == Path()


@pytest.mark.unit
def test_get_path_double_dot_returns_parent() -> None:
    """
    Given: ".." and a root
    When: calling get_path
    Then: returns the parent directory
    """
    # --- Act ---
    result = get_path("..", "/project/src")

    # --- Assert ---
    assert result == Path("/project")


@pytest.mark.unit
def test_get_path_triple_dot_returns_grandparent() -> None:
    """
    Given: "..." and a root
    When: calling get_path
    Then: returns the grandparent directory
    """
    # --- Act ---
    result = get_path("...", "/project/src")

    # --- Assert ---
    assert result == Path("/")


@pytest.mark.unit
def test_get_path_relative_with_root() -> None:
    """
    Given: a relative path and a root
    When: calling get_path
    Then: resolves correctly
    """
    # --- Act ---
    result = get_path("../foo", "/project/src")

    # --- Assert ---
    assert result == Path("/project/foo")


@pytest.mark.unit
def test_get_path_substring_lookup() -> None:
    """
    Given: a path that is a substring of root
    When: calling get_path
    Then: returns the prefix up to the first occurrence
    """
    # --- Act ---
    result = get_path("src", "/project/src/module.py")

    # --- Assert ---
    assert result == Path("/project/src")


@pytest.mark.unit
def test_get_path_walk_up_for_dirname() -> None:
    """
    Given: a directory name and a root
    When: calling get_path
    Then: walks up and returns the matching directory
    """
    # --- Act ---
    result = get_path("project", "/project/src")

    # --- Assert ---
    assert result == Path("/project")


@pytest.mark.unit
def test_get_path_raises_type_error_for_invalid_root() -> None:
    """
    Given: an invalid root type
    When: calling get_path
    Then: raises TypeError
    """
    # --- Act / Assert ---
    with pytest.raises(TypeError, match="root="):
        get_path("foo", 123)


@pytest.mark.unit
def test_get_path_raises_value_error_when_not_found(fake: Faker) -> None:
    """
    Given: a path that does not exist in root
    When: calling get_path
    Then: raises ValueError
    """
    # --- Arrange ---
    unmatched = fake.pystr(min_chars=8, max_chars=12)

    # --- Act / Assert ---
    with pytest.raises(ValueError, match="not found"):
        get_path(unmatched, "/project/src")


@pytest.mark.unit
def test_add_path_appends_resolved_path_to_sys_path(
    sys_path_snapshot: list[str],
    tmp_path: Path,
) -> None:
    """
    Given: a valid directory path
    When: calling add_path
    Then: appends the resolved path to sys.path
    """
    # --- Arrange ---
    subdir = tmp_path / "src"
    subdir.mkdir()

    # --- Act ---
    result = add_path(str(subdir))

    # --- Assert ---
    assert str(result) in sys.path


@pytest.mark.unit
def test_add_path_deduplicates_sys_path(
    sys_path_snapshot: list[str],
    tmp_path: Path,
) -> None:
    """
    Given: a path already in sys.path
    When: calling add_path twice
    Then: sys.path does not contain duplicates
    """
    # --- Arrange ---
    subdir = tmp_path / "src"
    subdir.mkdir()
    add_path(str(subdir))
    before_count = sys.path.count(str(subdir.resolve()))

    # --- Act ---
    add_path(str(subdir))

    # --- Assert ---
    assert sys.path.count(str(subdir.resolve())) == before_count


@pytest.mark.unit
def test_add_path_for_file_adds_parent_directory(
    sys_path_snapshot: list[str],
    tmp_path: Path,
) -> None:
    """
    Given: a file path
    When: calling add_path
    Then: adds the parent directory
    """
    # --- Arrange ---
    file_path = tmp_path / "module.py"
    file_path.write_text("# dummy")

    # --- Act ---
    result = add_path(str(file_path))

    # --- Assert ---
    assert str(tmp_path.resolve()) in sys.path
    assert result == tmp_path.resolve()


@pytest.mark.unit
def test_add_path_returns_path_object(
    sys_path_snapshot: list[str],
    tmp_path: Path,
) -> None:
    """
    Given: a valid path
    When: calling add_path
    Then: returns a Path instance
    """
    # --- Arrange ---
    subdir = tmp_path / "src"
    subdir.mkdir()

    # --- Act ---
    result = add_path(str(subdir))

    # --- Assert ---
    assert isinstance(result, Path)


# ------------------------------------------------------------------
# Security tests
# ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.security
def test_get_path_resolves_traversal_payload_safely(tmp_path: Path) -> None:
    """
    Given: a path-traversal payload
    When: calling get_path with a root
    Then: it resolves to a Path without filesystem escape.
    """
    # --- Arrange ---
    payload = "../../../etc/passwd"

    # --- Act ---
    result = get_path(payload, str(tmp_path))

    # --- Assert ---
    assert isinstance(result, Path)
    assert "/etc/passwd" not in str(result) or result.is_absolute()


@pytest.mark.unit
@pytest.mark.security
def test_add_path_resolves_traversal_without_error(
    sys_path_snapshot: list[str],
) -> None:
    """
    Given: a traversal path
    When: calling add_path
    Then: it resolves via pathlib and appends without exception.
    """
    # --- Arrange ---
    payload = "../.."

    # --- Act ---
    result = add_path(payload)

    # --- Assert ---
    assert isinstance(result, Path)
    assert str(result) in sys.path


@pytest.mark.unit
@pytest.mark.security
def test_required_rejects_malformed_path(fake: Faker) -> None:
    """
    Given: a malformed import path
    When: calling required
    Then: ImportError is raised.
    """
    # --- Arrange ---
    malformed = f"<{fake.pystr()}>alert(1)</{fake.pystr()}>"

    # --- Act / Assert ---
    with pytest.raises(ImportError):
        required(malformed)


@pytest.mark.unit
@pytest.mark.security
def test_required_logs_do_not_contain_pii(
    caplog: pytest.LogCaptureFixture,
    fake: Faker,
) -> None:
    """
    Given: a failed required import
    When: logging is captured
    Then: no PII appears in log records.
    """
    # --- Arrange ---
    invalid = f"{fake.pystr()}.{fake.pystr()}"

    # --- Act ---
    with (
        caplog.at_level(logging.WARNING, logger="kain.importer"),
        suppress(ImportError),
    ):
        required(invalid, throw=True)

    # --- Assert ---
    for record in caplog.records:
        assert "@" not in record.message
        assert ".com" not in record.message
        assert "token" not in record.message.lower()


# ------------------------------------------------------------------
# Annotation inference
# ------------------------------------------------------------------


class TestAnnotationInference:
    """GIVEN importer functions
    WHEN inspecting type hints
    THEN signatures match declared types.
    """

    def test_get_path_returns_path(self) -> None:
        """GIVEN get_path
        WHEN getting type hints
        THEN return type is Path.
        """
        # --- Act ---
        hints = get_type_hints(get_path)

        # --- Assert ---
        assert hints["return"] is Path

    def test_add_path_returns_path(self) -> None:
        """GIVEN add_path
        WHEN getting type hints
        THEN return type is Path.
        """
        # --- Act ---
        hints = get_type_hints(add_path)

        # --- Assert ---
        assert hints["return"] is Path

    def test_required_returns_object(self) -> None:
        """GIVEN required
        WHEN getting type hints
        THEN return type is object.
        """
        # --- Act ---
        hints = get_type_hints(required)

        # --- Assert ---
        assert hints["return"] is object

    def test_optional_returns_object(self) -> None:
        """GIVEN optional
        WHEN getting type hints
        THEN return type is object.
        """
        # --- Act ---
        hints = get_type_hints(optional)

        # --- Assert ---
        assert hints["return"] is object

    def test_get_module_returns_tuple(self) -> None:
        """GIVEN get_module
        WHEN getting type hints
        THEN return type is a tuple.
        """
        # --- Act ---
        hints = get_type_hints(get_module)

        # --- Assert ---
        assert "tuple" in str(hints["return"])


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestEdgeCases:
    """Paranoid edge-case coverage for importer."""

    def test_get_path_with_empty_string(self) -> None:
        """GIVEN empty string path
        WHEN calling get_path
        THEN ValueError is raised.
        """
        # --- Act / Assert ---
        with pytest.raises(ValueError, match="not found"):
            get_path("")

    def test_add_path_with_absolute_path(
        self,
        sys_path_snapshot: list[str],
        tmp_path: Path,
    ) -> None:
        """GIVEN an absolute path
        When: calling add_path
        Then: it is appended to sys.path.
        """
        # --- Arrange ---
        subdir = tmp_path / "abs"
        subdir.mkdir()

        # --- Act ---
        result = add_path(str(subdir))

        # --- Assert ---
        assert str(subdir) in sys.path
        assert result == subdir

    def test_optional_with_throw_true_raises(self, fake: Faker) -> None:
        """GIVEN optional with throw=True passed in kw
        WHEN import fails
        THEN ImportError is raised.
        """
        # --- Arrange ---
        invalid = f"{fake.pystr()}.{fake.pystr()}"

        # --- Act / Assert ---
        with pytest.raises(ImportError):
            optional(invalid, throw=True)
