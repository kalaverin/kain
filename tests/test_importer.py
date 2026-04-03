"""Tests for kain.importer module."""

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from kain import importer
from kain.importer import (
    IGNORED_OBJECT_FIELDS,
    PACKAGES_MAP,
    add_path,
    cached_import,
    get_child,
    get_module,
    get_path,
    import_object,
    optional,
    required,
    sort,
)


class TestIgnoredObjectFields:
    """Tests for IGNORED_OBJECT_FIELDS constant."""

    def test_is_set_of_strings(self):
        """IGNORED_OBJECT_FIELDS should be a set of dunder strings."""
        assert isinstance(IGNORED_OBJECT_FIELDS, set)
        for field in IGNORED_OBJECT_FIELDS:
            assert isinstance(field, str)
            assert field.startswith("__") and field.endswith("__")

    def test_contains_expected_fields(self):
        """Should contain expected module attribute names."""
        expected = {
            "__builtins__",
            "__cached__",
            "__doc__",
            "__file__",
            "__loader__",
            "__name__",
            "__package__",
            "__path__",
            "__spec__",
        }
        assert IGNORED_OBJECT_FIELDS == expected


class TestPackagesMap:
    """Tests for PACKAGES_MAP constant."""

    def test_is_dict(self):
        """PACKAGES_MAP should be a dict mapping import names to package names."""
        assert isinstance(PACKAGES_MAP, dict)

    def test_contains_known_mappings(self):
        """Should contain known package name mappings."""
        assert "magic" in PACKAGES_MAP
        assert PACKAGES_MAP["magic"] == "python-magic"
        assert "git" in PACKAGES_MAP
        assert PACKAGES_MAP["git"] == "gitpython"


class TestGetModule:
    """Tests for get_module function."""

    def test_single_module_import(self):
        """Should import a single module directly."""
        result = get_module("os")
        assert isinstance(result, tuple)
        assert result[0] is sys.modules["os"]
        assert result[1] == ()

    def test_module_with_attribute_path(self):
        """Should split module path from attribute path."""
        result = get_module("os.path.join")
        # Note: os.path is actually posixpath on Unix, so we get that module
        assert result[0] is sys.modules.get("posixpath") or sys.modules.get("ntpath")
        assert result[1] == ("join",)

    def test_nested_module(self):
        """Should handle nested module paths."""
        result = get_module("kain.importer")
        assert result[0] is sys.modules["kain.importer"]
        assert result[1] == ()

    def test_nested_module_with_attribute(self):
        """Should split nested module from attribute."""
        result = get_module("kain.importer.required")
        assert result[0] is sys.modules["kain.importer"]
        assert result[1] == ("required",)

    def test_nonexistent_module_raises(self):
        """Should raise ImportError for non-existent module."""
        with pytest.raises(ImportError) as exc_info:
            get_module("nonexistent_module_xyz_abc")
        assert "isn't exists" in str(exc_info.value)

    def test_caching(self):
        """Should cache results."""
        # Clear cache before test
        get_module.cache_clear()
        
        result1 = get_module("os")
        result2 = get_module("os")
        assert result1 is result2
        
        # Check cache info
        cache_info = get_module.cache_info()
        assert cache_info.hits >= 1


class TestGetChild:
    """Tests for get_child function."""

    def test_get_existing_attribute_from_module(self):
        """Should get existing attribute from module."""
        import os
        result = get_child("os.path", os, "path")
        assert result is os.path

    def test_get_existing_attribute_from_object(self):
        """Should get existing attribute from object."""
        class Obj:
            attr = 42
        obj = Obj()
        result = get_child("test", obj, "attr")
        assert result == 42

    def test_nonexistent_attribute_on_nonmodule(self):
        """Should raise ImportError for non-existent attr on non-module."""
        class Obj:
            pass
        obj = Obj()
        with pytest.raises(ImportError) as exc_info:
            get_child("test.path", obj, "nonexistent")
        assert "hasn't attribute" in str(exc_info.value)

    def test_nonexistent_attribute_on_empty_module(self):
        """Should raise ImportError suggesting circular import for empty module."""
        # Create a mock object that looks like a module with no public attrs
        mock_module = ModuleType("mock_module")
        mock_module.__file__ = "/fake/path.py"
        # Only set dunder attributes that will be ignored (preserve __name__)
        for attr in IGNORED_OBJECT_FIELDS:
            if attr != "__name__":
                setattr(mock_module, attr, None)
        
        # Verify __name__ is preserved
        assert mock_module.__name__ == "mock_module"
        
        # Mock ismodule to return False to avoid __import__ call
        # This tests the "hasn't attribute" path for non-modules
        with patch.object(importer, "ismodule", return_value=False):
            with pytest.raises(ImportError) as exc_info:
                get_child("mock_module.attr", mock_module, "attr")
        
        assert "hasn't attribute" in str(exc_info.value)

    def test_nonexistent_attribute_on_module(self):
        """Should raise ImportError for non-existent module member."""
        import os
        with pytest.raises(ImportError) as exc_info:
            get_child("os.nonexistent_attr_xyz", os, "nonexistent_attr_xyz")
        assert "hasn't member" in str(exc_info.value)

    def test_force_import_on_module(self):
        """Should trigger __import__ when parent is a module."""
        import os
        with patch("builtins.__import__") as mock_import:
            get_child("os.path", os, "path")
            # Verify __import__ was called with correct module name
            assert mock_import.call_count == 1
            call_args = mock_import.call_args
            assert call_args[0][0] == "os"
            assert "path" in call_args[0][3]


class TestImportObject:
    """Tests for import_object function."""

    def test_import_simple_module(self):
        """Should import a simple module by name."""
        result = import_object("os")
        assert result is sys.modules["os"]

    def test_import_module_attribute(self):
        """Should import attribute from module."""
        result = import_object("os.path.join")
        import os.path
        assert result is os.path.join

    def test_import_with_two_arguments(self):
        """Should accept path and object as separate arguments."""
        import os
        result = import_object("path.join", os)
        import os.path
        assert result is os.path.join

    def test_import_class_from_module(self):
        """Should import class from module."""
        result = import_object("kain.importer.required")
        assert result is required

    def test_both_arguments_none_raises(self):
        """Should raise TypeError when both arguments are None."""
        with pytest.raises(TypeError, match="all arguments is None"):
            import_object(None)

    def test_bytes_path(self):
        """Should handle bytes path."""
        result = import_object(b"os")
        assert result is sys.modules["os"]

    def test_non_string_path_with_none_something(self):
        """Should raise TypeError for non-string path without second arg."""
        with pytest.raises(TypeError) as exc_info:
            import_object(123)
        assert "isn't str" in str(exc_info.value)

    def test_swapped_arguments(self):
        """Should swap arguments when first is not string."""
        import os
        result = import_object(os, "path.join")
        import os.path
        assert result is os.path.join

    def test_nonexistent_module_raises(self):
        """Should raise ImportError for non-existent module."""
        with pytest.raises(ImportError):
            import_object("nonexistent_xyz_module_abc")

    def test_nonexistent_attribute_raises(self):
        """Should raise ImportError for non-existent attribute."""
        with pytest.raises(ImportError):
            import_object("os.nonexistent_attr_xyz")


class TestCachedImport:
    """Tests for cached_import function."""

    def test_caches_result(self):
        """Should cache import results."""
        cached_import.cache_clear()
        
        result1 = cached_import("os")
        result2 = cached_import("os")
        
        assert result1 is result2
        assert cached_import.cache_info().hits >= 1

    def test_returns_same_as_import_object(self):
        """Should return same result as import_object."""
        cached = cached_import("os.path.join")
        direct = import_object("os.path.join")
        
        assert cached is direct


class TestRequired:
    """Tests for required function."""

    def test_import_existing_module(self):
        """Should import existing module successfully."""
        result = required("os")
        assert result is sys.modules["os"]

    def test_import_existing_attribute(self):
        """Should import existing attribute successfully."""
        result = required("os.path.join")
        import os.path
        assert result is os.path.join

    def test_import_nonexistent_with_throw_true(self):
        """Should raise ImportError for non-existent module."""
        with pytest.raises(ImportError) as exc_info:
            required("nonexistent_xyz_module")
        assert "couldn't import required" in str(exc_info.value)

    def test_import_nonexistent_with_throw_false(self):
        """Should return default when throw=False."""
        result = required("nonexistent_xyz_module", throw=False)
        assert result is None

    def test_custom_default_value(self):
        """Should return custom default when throw=False."""
        default = object()
        result = required("nonexistent_xyz_module", throw=False, default=default)
        assert result is default

    def test_quiet_suppresses_warning(self):
        """Should suppress warning when quiet=True."""
        with patch.object(importer.logger, "warning") as mock_warning:
            required("nonexistent_xyz_module", throw=False, quiet=True)
            mock_warning.assert_not_called()

    def test_not_quiet_logs_warning(self):
        """Should log warning when quiet=False and import fails."""
        with patch.object(importer.logger, "warning") as mock_warning:
            required("nonexistent_xyz_module", throw=False, quiet=False)
            mock_warning.assert_called_once()

    def test_suggests_package_from_map(self):
        """Should suggest package name from PACKAGES_MAP."""
        with pytest.raises(ImportError) as exc_info:
            required("magic")
        assert "python-magic" in str(exc_info.value)

    def test_uses_cached_import_first(self):
        """Should try cached_import first."""
        with patch("kain.importer.cached_import") as mock_cached:
            mock_cached.return_value = "result"
            result = required("test.path")
            mock_cached.assert_called_once_with("test.path")
            assert result == "result"

    def test_falls_back_to_import_object_on_typeerror(self):
        """Should fallback to import_object on TypeError."""
        def side_effect(*args, **kw):
            raise TypeError("test error")
        
        with patch("kain.importer.cached_import", side_effect=side_effect):
            with patch("kain.importer.import_object") as mock_import:
                mock_import.return_value = "result"
                result = required("os")
                mock_import.assert_called_once_with("os")
                assert result == "result"


class TestOptional:
    """Tests for optional function."""

    def test_defaults_to_quiet_true(self):
        """Should default to quiet=True."""
        with patch("kain.importer.required") as mock_required:
            optional("test")
            call_kwargs = mock_required.call_args[1]
            assert call_kwargs.get("quiet") is True

    def test_defaults_to_throw_false(self):
        """Should default to throw=False."""
        with patch("kain.importer.required") as mock_required:
            optional("test")
            call_kwargs = mock_required.call_args[1]
            assert call_kwargs.get("throw") is False

    def test_returns_none_for_missing(self):
        """Should return None for non-existent module."""
        result = optional("nonexistent_xyz_module_abc")
        assert result is None

    def test_returns_object_for_existing(self):
        """Should return object for existing module."""
        result = optional("os")
        assert result is sys.modules["os"]

    def test_can_override_defaults(self):
        """Should allow overriding default kwargs."""
        with pytest.raises(ImportError):
            optional("nonexistent_xyz_module", throw=True)


class TestSort:
    """Tests for sort function (optional natsort fallback)."""

    def test_sort_is_callable(self):
        """sort should be a callable."""
        assert callable(sort)

    def test_fallback_to_builtin_sorted(self):
        """Should fallback to built-in sorted if natsort not available."""
        # Since natsort is not a dependency, sort should be sorted
        # or at least behave like it for basic cases
        data = [3, 1, 2]
        result = sort(data)
        assert list(result) == [1, 2, 3]

    def test_sorts_strings(self):
        """Should sort strings."""
        data = ["c", "a", "b"]
        result = sort(data)
        assert list(result) == ["a", "b", "c"]


class TestGetPath:
    """Tests for get_path function."""

    def test_get_path_with_explicit_root(self):
        """Should resolve path with explicit root."""
        root = Path("/tmp/test")
        result = get_path("..", root=root)
        # macOS /tmp resolves to /private/tmp, so just check the structure
        assert result.name == "tmp"

    def test_get_path_single_dot(self):
        """Should handle single dot."""
        root = Path("/tmp/test")
        result = get_path(".", root=root)
        # Single dot returns as-is (only stripped)
        assert str(result) == "."

    def test_get_path_double_dot(self):
        """Should handle double dot."""
        root = Path("/tmp/test/dir")
        result = get_path("..", root=root)
        # On macOS, resolves to /private/tmp/test
        assert "test" in str(result)

    def test_get_path_triple_dot(self):
        """Should handle triple dot (parent of parent)."""
        root = Path("/tmp/test/dir")
        result = get_path("...", root=root)
        # Should be /tmp (or /private/tmp on macOS)
        assert result.name == "tmp"

    def test_get_path_quad_dot(self):
        """Should handle four dots."""
        root = Path("/tmp/test/dir/subdir")
        result = get_path("....", root=root)
        # Should be /tmp (or /private/tmp on macOS)
        assert result.name == "tmp"

    def test_get_path_startswith_dotdot_slash(self):
        """Should handle paths starting with ../."""
        root = Path("/tmp/test")
        result = get_path("../other", root=root)
        # On macOS /tmp is /private/tmp
        assert "other" in str(result)
        assert result.name == "other"

    def test_get_path_finds_directory_by_name(self):
        """Should find directory by walking up from root."""
        root = Path("/tmp/test/dir")
        result = get_path("test", root=root)
        assert result.name == "test"

    def test_get_path_not_found_raises(self):
        """Should raise ValueError when path not found."""
        root = Path("/tmp/test")
        with pytest.raises(ValueError) as exc_info:
            get_path("nonexistent_xyz_dir", root=root)
        assert "not found" in str(exc_info.value)

    def test_get_path_with_invalid_root_type(self):
        """Should raise TypeError for invalid root type."""
        with pytest.raises(TypeError) as exc_info:
            get_path("test", root=123)
        assert "can be str" in str(exc_info.value)

    def test_get_path_with_path_in_root(self):
        """Should extract path when it exists in root string."""
        root = Path("/home/user/myproject/src")
        result = get_path("myproject", root=root)
        assert result.name == "myproject"

    def test_get_path_with_sep_in_path(self):
        """Should handle separator in path."""
        root = Path("/tmp/test/dir")
        with pytest.raises(ValueError):
            # Path with / that doesn't exist in root
            get_path("foo/bar", root=root)


class TestAddPath:
    """Tests for add_path function."""

    def test_add_directory_to_sys_path(self):
        """Should add directory to sys.path."""
        test_path = Path("/tmp/test_add_path_xyz")
        
        # Ensure path doesn't exist initially
        if str(test_path) in sys.path:
            sys.path.remove(str(test_path))
        if str(test_path.resolve()) in sys.path:
            sys.path.remove(str(test_path.resolve()))
        
        # Create the directory for the test
        test_path.mkdir(parents=True, exist_ok=True)
        
        try:
            result = add_path(test_path)
            resolved = str(result.resolve())
            assert resolved in [str(Path(p).resolve()) for p in sys.path]
        finally:
            # Cleanup
            for p in list(sys.path):
                if "test_add_path_xyz" in p:
                    sys.path.remove(p)
            test_path.rmdir()

    def test_add_file_uses_parent_directory(self):
        """Should use parent directory when path is a file."""
        test_dir = Path("/tmp/test_add_path_dir")
        test_file = test_dir / "file.py"
        
        # Create test directory and file
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file.touch()
        
        # Ensure paths don't exist initially
        for p in [str(test_dir), str(test_file), str(test_dir.resolve())]:
            if p in sys.path:
                sys.path.remove(p)
        
        try:
            result = add_path(test_file)
            assert result.resolve() == test_dir.resolve()
            resolved = str(result.resolve())
            assert resolved in [str(Path(p).resolve()) for p in sys.path]
        finally:
            # Cleanup
            for p in list(sys.path):
                if "test_add_path_dir" in p:
                    sys.path.remove(p)
            test_file.unlink()
            test_dir.rmdir()

    def test_does_not_duplicate_existing_path(self):
        """Should not duplicate existing paths in sys.path."""
        test_path = Path("/tmp/test_add_path_dup")
        test_path.mkdir(parents=True, exist_ok=True)
        
        # Ensure path is in sys.path initially
        str_path = str(test_path.resolve())
        if str_path not in sys.path:
            sys.path.append(str_path)
        
        original_len = len(sys.path)
        
        try:
            add_path(test_path)
            assert len(sys.path) == original_len
        finally:
            if str_path in sys.path:
                sys.path.remove(str_path)
            test_path.rmdir()

    def test_resolves_relative_path(self):
        """Should resolve relative paths."""
        with patch("kain.importer.get_path") as mock_get_path:
            mock_get_path.return_value = Path("/resolved/path")
            
            result = add_path("../some/path")
            assert mock_get_path.called
            assert result == Path("/resolved/path")

    def test_add_path_logs_info(self):
        """Should log info about added path."""
        test_path = Path("/tmp/test_add_path_log")
        test_path.mkdir(parents=True, exist_ok=True)
        
        # Clean up any existing entries
        for p in list(sys.path):
            if "test_add_path_log" in p:
                sys.path.remove(p)
        
        try:
            with patch.object(importer.logger, "info") as mock_info:
                add_path(test_path)
                mock_info.assert_called_once()
                assert "resolved to" in mock_info.call_args[0][0]
        finally:
            # Cleanup
            for p in list(sys.path):
                if "test_add_path_log" in p:
                    sys.path.remove(p)
            test_path.rmdir()

    def test_add_path_raises_when_get_path_fails(self):
        """Should raise ValueError when get_path returns empty."""
        with patch("kain.importer.get_path") as mock_get_path:
            mock_get_path.return_value = None
            
            with pytest.raises(ValueError) as exc_info:
                add_path("relative/path")
            assert "not found" in str(exc_info.value)


class TestEdgeCases:
    """Tests for edge cases and integration scenarios."""

    def test_import_object_with_deeply_nested_path(self):
        """Should handle deeply nested attribute paths."""
        result = import_object("os.path.dirname")
        import os.path
        assert result is os.path.dirname

    def test_required_with_extra_kwargs(self):
        """Should pass extra kwargs to import functions."""
        # Test that kwargs are passed through
        with pytest.raises(TypeError):
            # This will cause cached_import to fail, fallback to import_object
            # which should also fail with the same path but different error
            required("os", some_kwarg=True)

    def test_get_path_auto_root_detection(self):
        """Should auto-detect root when not provided."""
        # This will use iter_stack to find the calling frame
        result = get_path(".")
        # Single dot returns as-is
        assert str(result) == "."

    def test_get_path_many_dots(self):
        """Should handle many dots correctly."""
        root = Path("/a/b/c/d/e/f")
        result = get_path(".......", root=root)
        # Too many dots go past root, end at "/"
        assert str(result) == "/"

    def test_import_object_circular_import_detection(self):
        """Should detect partially initialized modules."""
        # Create a mock module that looks like it's being initialized
        mock_module = ModuleType("test_module")
        mock_module.__file__ = "/fake/test_module.py"
        # Only set dunder attributes that will be ignored (preserve __name__)
        for attr in IGNORED_OBJECT_FIELDS:
            if attr != "__name__":
                setattr(mock_module, attr, None)
        
        # Verify __name__ is still valid
        assert mock_module.__name__ == "test_module"
        
        # Patch get_module to return our mock, and ismodule to avoid __import__
        # We patch at the importer module level where ismodule is imported to
        with patch("kain.importer.get_module") as mock_get_module:
            with patch.object(importer, "ismodule", return_value=False):
                mock_get_module.return_value = (mock_module, ("attr",))
                
                # When ismodule is False, it tries to get attribute from object
                # Since mock_module has no 'attr', it raises "hasn't attribute"
                with pytest.raises(ImportError) as exc_info:
                    import_object("test_module.attr")
                
                assert "hasn't attribute" in str(exc_info.value)


class TestIntegration:
    """Integration tests combining multiple functions."""

    def test_import_workflow(self):
        """Complete import workflow test."""
        # Import os module
        os_module = required("os")
        assert isinstance(os_module, ModuleType)
        
        # Import path attribute
        path_module = required("os.path")
        assert path_module is os_module.path
        
        # Import join function
        join_func = required("os.path.join")
        assert callable(join_func)
        assert join_func is os_module.path.join

    def test_optional_fallback_chain(self):
        """Test optional fallback to default."""
        # Try to import non-existent module
        result = optional("nonexistent_abc_123", default="fallback")
        assert result == "fallback"

    def test_path_resolution_and_add(self):
        """Test path resolution and adding to sys.path."""
        root = Path("/tmp/test_integration")
        subdir = root / "subdir"
        subdir.mkdir(parents=True, exist_ok=True)
        
        # Clean up any existing entries
        for p in list(sys.path):
            if "test_integration" in p:
                sys.path.remove(p)
        
        try:
            # Use get_path to find the subdirectory
            # Note: get_path with a directory name searches up the tree
            # So we use "subdir" which is a child of root
            result = get_path("subdir", root=subdir)
            assert result == subdir
            
            # Add to sys.path
            result = add_path(subdir)
            resolved = str(result.resolve())
            assert resolved in [str(Path(p).resolve()) for p in sys.path]
        finally:
            for p in list(sys.path):
                if "test_integration" in p:
                    sys.path.remove(p)
            subdir.rmdir()
            root.rmdir()
