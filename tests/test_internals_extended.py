"""Extended tests for kain.internals with focus on typing variants.

These tests exercise edge cases, typing constructs, and parameter
combinations not fully covered by the base test suite.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections import ChainMap, OrderedDict, defaultdict, deque
from collections.abc import Iterator
from functools import partial
from types import MappingProxyType, UnionType
from typing import (
    Any,
    ClassVar,
    Final,
    Generic,
    Literal,
    NamedTuple,
    NewType,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
)

import pytest

from kain.internals import (
    class_of,
    format_args_and_keywords,
    get_attr,
    get_module,
    get_module_name,
    get_mro,
    get_owner,
    is_callable,
    is_collection,
    is_from_builtin,
    is_from_primitive,
    is_imported_module,
    is_internal,
    is_iterable,
    is_mapping,
    is_primitive,
    is_subclass,
    isclass,
    iter_inheritance,
    iter_stack,
    just_value,
    object_name,
    pretty_module,
    simple_repr,
    source_file,
    to_ascii,
    to_bytes,
    unique,
    who_is,
)


class TestClassOfTypingVariants:
    """Runtime type-shape checks for ``class_of``."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (42, int),
            ("s", str),
            (None, type(None)),
            (int, int),
            (list[int], types.GenericAlias),
            (int | str, UnionType),
            (..., type(...)),
            (NotImplemented, type(NotImplemented)),
            (TypeVar("T"), TypeVar),
        ],
    )
    def test_class_of(self, obj: Any, expected: type[Any]) -> None:
        assert class_of(obj) is expected


class TestIsCallableTypingVariants:
    """Callable recognition beyond plain functions."""

    @pytest.mark.parametrize(
        "obj",
        [
            lambda: 1,
            print,
            len,
            partial(max, 10),
            object().__str__,
        ],
    )
    def test_callable_true(self, obj: Any) -> None:
        assert is_callable(obj) is True

    @pytest.mark.parametrize(
        "obj",
        [
            1,
            "hello",
            None,
            object(),
            [1, 2, 3],
        ],
    )
    def test_callable_false(self, obj: Any) -> None:
        assert is_callable(obj) is False

    def test_callable_custom_with_dunder_call(self) -> None:
        class Caller:
            def __call__(self) -> int:
                return 1

        assert is_callable(Caller()) is True


class TestIsCollectionTypingVariants:
    """Collection detection for stdlib and duck-typed objects."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ([1, 2], True),
            ((1, 2), True),
            ({"a": 1}, True),
            (deque([1, 2]), True),
            ({1, 2}, False),
            ("abc", False),
            (b"abc", False),
            (defaultdict(list), True),
            (OrderedDict(), True),
            (ChainMap(), True),
            (MappingProxyType({"x": 1}), True),
        ],
    )
    def test_collection(self, obj: Any, expected: bool) -> None:
        assert is_collection(obj) is expected

    def test_collection_duck_typed(self) -> None:
        class Duck:
            def __getitem__(self, key: Any) -> Any:
                return key

            def __setitem__(self, key: Any, value: Any) -> None:
                pass

            def __delitem__(self, key: Any) -> None:
                pass

        assert is_collection(Duck()) is True

    def test_collection_partial_duck(self) -> None:
        class PartialDuck:
            def __getitem__(self, key: Any) -> Any:
                return key

        assert is_collection(PartialDuck()) is False


class TestIsIterableTypingVariants:
    """Iterable recognition for generators, iterators, and duck types."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ([1, 2], True),
            ("a", True),
            ({"a": 1}, True),
            ((x for x in range(3)), True),
            (iter([1, 2]), True),
            (1, False),
            (None, False),
            (object(), False),
        ],
    )
    def test_iterable(self, obj: Any, expected: bool) -> None:
        assert is_iterable(obj) is expected

    def test_iterable_duck(self) -> None:
        class DuckIterable:
            def __iter__(self) -> iter[Any]:
                return iter([])

        assert is_iterable(DuckIterable()) is True


class TestIsMappingTypingVariants:
    """Mapping detection across stdlib specialisations."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ({"a": 1}, True),
            (defaultdict(list), True),
            (OrderedDict(), True),
            (ChainMap(), True),
            (MappingProxyType({"x": 1}), True),
            ([1, 2], False),
            ((1, 2), False),
            (set(), False),
            ("abc", False),
        ],
    )
    def test_mapping(self, obj: Any, expected: bool) -> None:
        assert is_mapping(obj) is expected

        class MyDict(dict):
            pass

        assert is_mapping(MyDict()) is True


class TestIsPrimitiveTypingVariants:
    """Primitive checks covering every member of ``Builtins``."""

    @pytest.mark.parametrize(
        "obj",
        [
            True,
            False,
            None,
            1,
            1.5,
            1j,
            "hello",
            b"hello",
            [1, 2],
            (1, 2),
            {"a": 1},
            {1, 2},
            deque([1, 2]),
            bytearray(b"x"),
        ],
    )
    def test_primitive_true(self, obj: Any) -> None:
        assert is_primitive(obj) is True

    @pytest.mark.parametrize(
        "obj",
        [
            object(),
            lambda: 1,
            type("Custom", (), {}),
        ],
    )
    def test_primitive_false(self, obj: Any) -> None:
        assert is_primitive(obj) is False


class TestIsFromPrimitiveAndBuiltin:
    """Boundary between primitive, builtin, and custom objects."""

    def test_from_primitive(self) -> None:
        assert is_from_primitive(1) is True
        assert is_from_primitive(1j) is True
        assert is_from_primitive(b"x") is True
        assert is_from_primitive([1]) is False

    def test_from_builtin(self) -> None:
        assert is_from_builtin([1]) is True
        assert is_from_builtin(deque([1])) is True
        assert is_from_builtin(bytearray(b"x")) is True
        assert is_from_builtin(object()) is False


class TestIsInternalTypingVariants:
    """Internal (stdlib / builtins) detection for typing artefacts."""

    @pytest.mark.parametrize(
        "obj",
        [
            1,
            int,
            print,
            len,
            os,
            sys,
            json,
            json.dumps,
            Any,
            UnionType,
            list[int],
        ],
    )
    def test_internal_true(self, obj: Any) -> None:
        assert is_internal(obj) is True

    @pytest.mark.parametrize(
        "obj",
        [
            type("Custom", (), {}),
            lambda: 1,
        ],
    )
    def test_internal_false(self, obj: Any) -> None:
        assert is_internal(obj) is False


class TestIsSubclassTypingVariants:
    """``is_subclass`` exercised against the full typing zoo."""

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            # Basic
            (1, int, True),
            (1, str, False),
            (None, type(None), True),
            (None, object, True),
            # Any
            (1, Any, True),
            # Union via pipe (types.UnionType)
            (1, int | str, True),
            (1, str | bytes, False),
            (None, int | str | None, True),
            (None, int | str, False),
            # Union via typing.Union
            (1, Union[int, str], True),
            (1, Union[str, bytes], False),
            # Optional
            (1, Optional[int], True),
            (None, Optional[int], True),
            # GenericAlias
            ([], list[int], True),
            ([], dict[str, int], False),
            ({}, dict[str, int], True),
            # typing.List
            ([], list[int], True),
            ([], list[str], True),  # origin match
            # None types
            (1, None, False),
        ],
    )
    def test_subclass(
        self,
        obj: Any,
        types: Any,
        expected: bool,
    ) -> None:
        assert is_subclass(obj, types) is expected

    def test_subclass_generic_alias_classes(self) -> None:
        """GenericAlias itself should be recognised via origin."""
        alias = dict[str, int]
        assert is_subclass({}, alias) is True
        assert is_subclass([], alias) is False


class TestGetModuleAndNameTypingVariants:
    """Module resolution for values that carry typing metadata."""

    def test_get_module_typing_specials(self) -> None:
        assert get_module(Any).__name__ == "typing"
        assert get_module(UnionType).__name__ == "types"

    def test_get_module_name_typing_specials(self) -> None:
        assert get_module_name(Any) == "typing"
        assert get_module_name(UnionType) == "types"

    def test_get_module_none_for_weird(self) -> None:
        class NoModule:
            pass

        instance = NoModule()
        # Explicitly wipe __module__ to see fallback behaviour
        instance.__module__ = None  # type: ignore[attr-defined]
        mod = get_module(instance)
        assert mod is None or mod.__name__ is not None


class TestObjectNameTypingVariants:
    """``object_name`` for typing constructs and edge cases."""

    @pytest.mark.parametrize(
        ("obj", "full", "expected_tail"),
        [
            (Any, True, "typing.Any"),
            (UnionType, True, "types.UnionType"),
            (list[int], True, "list"),
            (Literal[1], True, "typing.Literal"),
            (TypeVar("T"), False, "T"),
            (ParamSpec("P"), False, "P"),
            (Protocol, True, "typing.Protocol"),
            (Generic, True, "typing.Generic"),
        ],
    )
    def test_object_name(
        self,
        obj: Any,
        full: bool,
        expected_tail: str,
    ) -> None:
        name = object_name(obj, full=full)
        assert name == expected_tail or name.endswith(expected_tail)

    def test_object_name_generic_nested(self) -> None:
        alias = dict[str, list[int]]
        assert object_name(alias) == "dict"

    def test_object_name_builtin_bound_method(self) -> None:
        assert object_name([].append).endswith("list.append")


class TestPrettyModuleTypingVariants:
    """Module path extraction for typing and stdlib objects."""

    def test_pretty_module_stdlib(self) -> None:
        assert pretty_module(os.path.join) in (
            "os.path",
            "posixpath",
            "ntpath",
        )

    def test_pretty_module_typing(self) -> None:
        assert pretty_module(Any) == "typing"

    def test_pretty_module_builtin(self) -> None:
        # who_is(int) == "int" -> rsplit gives ["int"]
        assert pretty_module(int) == "int"


class TestSourceFileVariants:
    """Source file resolution with filters and templates."""

    def test_source_file_builtin_none(self) -> None:
        assert source_file(1) is None

    def test_source_file_custom_exclude_stdlib(self) -> None:
        class Custom:
            pass

        path = source_file(Custom, exclude_stdlib=False)
        assert path is None or path.endswith(".py")

    def test_source_file_with_template(self) -> None:
        class Custom:
            pass

        path = source_file(Custom, template="file://%s", exclude_stdlib=False)
        assert path is None or path.startswith("file://")


class TestJustValueVariants:
    """String representation combining type name and value."""

    def test_just_value_instance(self) -> None:
        assert just_value(42) == "(int)42"

    def test_just_value_class(self) -> None:
        assert just_value(int) == "(int)"

    def test_just_value_addr(self) -> None:
        s = just_value("hello", addr=True)
        assert "(str#" in s
        assert "#" in s


class TestIsImportedModuleVariants:
    """Module import state detection with dotted names."""

    @pytest.mark.parametrize(
        "name",
        [
            "os",
            "os.path",
            "sys",
            "collections.abc",
            "typing",
        ],
    )
    def test_imported_true(self, name: str) -> None:
        assert is_imported_module(name) is True

    @pytest.mark.parametrize(
        "name",
        [
            "definitely_not_imported_module_12345",
            "os.definitely_not_imported_submodule_12345",
        ],
    )
    def test_imported_false(self, name: str) -> None:
        assert is_imported_module(name) is False

    def test_imported_partial_match(self) -> None:
        # "os" is imported, "os.path" is imported, but "os.path.X" may not be
        assert is_imported_module("os.path.basename") is True  # nested enough


class TestGetMroVariants:
    """``get_mro`` with mapping and string gluing."""

    def test_get_mro_basic(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B) == (A,)

    def test_get_mro_func_and_glue(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B, func=lambda x: x.__name__) == ("A",)
        assert get_mro(B, func=lambda x: x.__name__, glue=" -> ") == "A"

    def test_get_mro_with_stdlib_exclusion(self) -> None:
        class A(Exception):
            pass

        mro = get_mro(A, exclude_stdlib=False)
        assert Exception in mro


class TestSimpleReprVariants:
    """Simplified repr for logging-like formatting."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (None, None),
            (True, True),
            (False, False),
            ("hello", "hello"),
            (42, "42"),
            (3.14, "3.14"),
        ],
    )
    def test_simple_repr(self, obj: Any, expected: Any) -> None:
        assert simple_repr(obj) == expected

    def test_simple_repr_custom(self) -> None:
        class Foo:
            def __repr__(self) -> str:
                return "<Foo>"

        out = simple_repr(Foo())
        assert "Foo" in out


class TestFormatArgsAndKeywordsVariants:
    """Positional and keyword argument formatting."""

    def test_format_empty(self) -> None:
        assert format_args_and_keywords() == ""

    def test_format_args_only(self) -> None:
        assert format_args_and_keywords(1, "a") == "'1', 'a'"

    def test_format_kwargs_only(self) -> None:
        assert format_args_and_keywords(x=1, y="a") == "x=1, y=a"

    def test_format_mixed(self) -> None:
        out = format_args_and_keywords(1, None, x=[1, 2])
        assert "'1'" in out
        assert "None" in out
        assert "x=" in out

    def test_format_nested(self) -> None:
        out = format_args_and_keywords({"a": 1})
        assert "dict" in out or "'a'" in out


class TestIterInheritanceVariants:
    """MRO traversal with include / exclude predicates."""

    def test_iter_inheritance_callable_include(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        result = list(iter_inheritance(C, include=lambda cls: cls is A))
        assert result == [A]

    def test_iter_inheritance_callable_exclude(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, exclude=lambda cls: cls is A))
        assert A not in result

    def test_iter_inheritance_include_tuple(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, include=(A,)))
        assert result == [A]

    def test_iter_inheritance_exclude_tuple(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, exclude=(A,)))
        assert result == []

    def test_iter_inheritance_no_exclude_stdlib(self) -> None:
        class A(Exception):
            pass

        result = list(iter_inheritance(A, exclude_stdlib=False))
        assert Exception in result

    def test_iter_inheritance_reverse(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        result = list(iter_inheritance(C, reverse=True, exclude_stdlib=False))
        # Should be object-last order reversed; object is always sliced off
        assert result[0] is A


class TestIterStackVariants:
    """Call-stack introspection with attribute extraction."""

    def test_iter_stack_single_attr(self) -> None:
        result = list(iter_stack(3))
        assert any(
            "test_iter_stack_single_attr" in str(item) for item in result
        )

    def test_iter_stack_multi_attr(self) -> None:
        result = list(iter_stack(0, 3))
        for pair in result:
            assert isinstance(pair, tuple)
            assert len(pair) == 2


class TestGetOwnerAndAttrVariants:
    """Attribute provenance across the inheritance chain."""

    def test_get_owner_with_index(self) -> None:
        class A:
            x = 1

        class B(A):
            x = 2

        class C(B):
            pass

        assert get_owner(C, "x") is B
        assert get_owner(C, "x", index=1) is A

    def test_get_owner_missing(self) -> None:
        class A:
            pass

        assert get_owner(A, "missing") is None

    def test_get_attr_with_index(self) -> None:
        class A:
            x = 1

        class B(A):
            x = 2

        class C(B):
            pass

        assert get_attr(C, "x") == 2
        assert get_attr(C, "x", index=1) == 1

    def test_get_attr_default(self) -> None:
        class A:
            pass

        assert get_attr(A, "nope", default="fallback") == "fallback"


class TestToAsciiAndBytesVariants:
    """Byte / string coercion with charset support."""

    @pytest.mark.parametrize(
        ("inp", "charset", "expected"),
        [
            ("hello", None, "hello"),
            (b"hello", None, "hello"),
            ("café", "utf-8", "café"),
            ("café".encode(), "utf-8", "café"),
        ],
    )
    def test_to_ascii(
        self,
        inp: bytes | str,
        charset: str | None,
        expected: str,
    ) -> None:
        kw: dict[str, str] = {}
        if charset:
            kw["charset"] = charset
        assert to_ascii(inp, **kw) == expected

    @pytest.mark.parametrize(
        ("inp", "charset", "expected"),
        [
            ("hello", None, b"hello"),
            (b"hello", None, b"hello"),
            ("café", "utf-8", "café".encode()),
        ],
    )
    def test_to_bytes(
        self,
        inp: bytes | str,
        charset: str | None,
        expected: bytes,
    ) -> None:
        kw: dict[str, str] = {}
        if charset:
            kw["charset"] = charset
        assert to_bytes(inp, **kw) == expected

    def test_to_ascii_invalid(self) -> None:
        with pytest.raises(TypeError):
            to_ascii(123)

    def test_to_bytes_invalid(self) -> None:
        with pytest.raises(TypeError):
            to_bytes(123)


class TestUniqueVariants:
    """Unique filtering for sequences and mappings with key funcs."""

    def test_unique_basic(self) -> None:
        assert list(unique([1, 2, 2, 3, 1])) == [1, 2, 3]

    def test_unique_with_key(self) -> None:
        assert list(unique([1, 2, 2, 3, 1], key=lambda x: x % 2)) == [1, 2]

    def test_unique_dict(self) -> None:
        result = list(unique({"a": 1, "b": 2}))
        assert result == [("a", 1), ("b", 2)]

    def test_unique_include(self) -> None:
        assert list(unique([1, 2, 3, 4], include=[1, 2])) == [1, 2]

    def test_unique_exclude(self) -> None:
        assert list(unique([1, 2, 3, 4], exclude=[2])) == [1, 3, 4]

    def test_unique_key_with_include(self) -> None:
        data = [1, 2, 3, 4]
        result = list(unique(data, key=lambda x: x % 2, include=[1]))
        # 1 and 3 share the same key (1), so only the first is kept.
        assert result == [1]

    def test_unique_key_with_exclude(self) -> None:
        data = [1, 2, 3, 4]
        result = list(unique(data, key=lambda x: x % 2, exclude=[0]))
        # 1 and 3 share the same key (1), so only the first is kept.
        assert result == [1]

    def test_unique_mapping_with_key(self) -> None:
        mapping = {"a": 1, "b": 2, "c": 3}
        result = list(unique(mapping, key=lambda k: k))
        assert ("a", 1) in result
        assert ("b", 2) in result

    def test_unique_empty(self) -> None:
        assert list(unique([])) == []

    def test_unique_generator(self) -> None:
        gen = (x for x in [1, 1, 2, 2, 3])
        assert list(unique(gen)) == [1, 2, 3]


class TestWhoIsVariants:
    """``who_is`` with caching and address suffixes."""

    def test_who_is_caches(self) -> None:
        class Dummy:
            pass

        d = Dummy()
        name1 = who_is(d)
        name2 = who_is(d)
        assert name1 == name2
        assert d.__dict__.get("__name_full__") == name1

    def test_who_is_short_caches(self) -> None:
        class Dummy:
            pass

        d = Dummy()
        name1 = who_is(d, full=False)
        name2 = who_is(d, full=False)
        assert name1 == name2
        assert d.__dict__.get("__name_short__") == name1

    def test_who_is_addr(self) -> None:
        s = who_is("x", addr=True)
        assert s.startswith("str#")


class TestGenericAliasAdvanced:
    """Advanced ``GenericAlias`` behaviours."""

    @pytest.mark.parametrize(
        ("alias", "obj", "expected"),
        [
            (list[int], [1, 2], True),
            (list[int], (1, 2), False),
            (dict[str, int], {"a": 1}, True),
            (dict[str, int], [], False),
            (set[int], {1, 2}, True),
            (set[int], [1, 2], False),
            (tuple[int, ...], (1, 2), True),
            (tuple[int, ...], [1, 2], False),
        ],
    )
    def test_is_subclass_generic_alias(
        self,
        alias: Any,
        obj: Any,
        expected: bool,
    ) -> None:
        assert is_subclass(obj, alias) is expected

    def test_class_of_generic_alias(self) -> None:
        assert class_of(list[int]) is types.GenericAlias
        assert class_of(dict[str, int]) is types.GenericAlias


class TestUnionTypeAdvanced:
    """``UnionType`` (pipe syntax) vs ``typing.Union``."""

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            (1, int | str, True),
            (None, int | str | None, True),
            (None, int | str, False),
            ("x", str | int | bytes, True),
            (3.14, int | str, False),
        ],
    )
    def test_union_type(self, obj: Any, types: Any, expected: bool) -> None:
        assert is_subclass(obj, types) is expected

    def test_union_type_vs_typing_union(self) -> None:
        assert is_subclass(1, Union[int, str]) is True
        assert is_subclass(1, int | str) is True
        assert is_subclass(None, Optional[int]) is True
        assert is_subclass(None, int | None) is True


class TestOptionalAdvanced:
    """Advanced ``Optional`` checks."""

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            (1, Optional[int], True),
            (None, Optional[int], True),
            ("x", Optional[int], False),
            (0, Optional[int], True),
            ("", Optional[str], True),
            (None, Optional[str], True),
        ],
    )
    def test_optional(self, obj: Any, types: Any, expected: bool) -> None:
        assert is_subclass(obj, types) is expected


class TestTypeVarBounds:
    """``TypeVar`` with bounds and constraints."""

    def test_typevar_basic(self) -> None:
        T = TypeVar("T")
        assert is_subclass(1, T) is False
        assert class_of(T) is TypeVar

    def test_typevar_bound(self) -> None:
        T = TypeVar("T", bound=int)
        assert is_subclass(1, T) is False

    def test_typevar_constrained(self) -> None:
        T = TypeVar("T", int, str)
        assert is_subclass(1, T) is False


class TestProtocolSubclasses:
    """Protocol subclass checks."""

    def test_protocol_class_of(self) -> None:
        class MyProto(Protocol):
            pass

        assert isclass(MyProto) is True

    def test_protocol_subclass_detection(self) -> None:
        class Drawable(Protocol):
            def draw(self) -> None: ...

        class Circle:
            def draw(self) -> None:
                pass

        assert (
            is_subclass(Circle(), Drawable) is False
        )  # Protocol not handled specially
        assert (
            is_subclass(Circle(), Drawable) is False
        )  # Protocol not handled specially

    def test_protocol_inheritance(self) -> None:
        class Base(Protocol):
            pass

        class Derived(Base, Protocol):
            pass

        result = list(iter_inheritance(Derived, exclude_stdlib=False))
        assert Base in result


class TestIsSubclassExoticTypes:
    """``is_subclass`` with exotic typing constructs."""

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            ([1], list, True),
            ({"a": 1}, dict, True),
            (42, Any, True),
            (None, type(None), True),
        ],
    )
    def test_exotic_types(self, obj: Any, types: Any, expected: bool) -> None:
        assert is_subclass(obj, types) is expected

    def test_is_subclass_literal_graceful(self) -> None:
        assert is_subclass(1, Literal[1, 2, 3]) is False
        assert is_subclass("a", Literal["a", "b"]) is False

    def test_is_subclass_named_tuple(self) -> None:
        class Point(NamedTuple):
            x: int
            y: int

        assert is_subclass(Point(1, 2), tuple) is True
        assert class_of(Point(1, 2)) is Point

    def test_is_subclass_newtype(self) -> None:
        UserId = NewType("UserId", int)
        assert is_subclass(UserId(1), int) is True
        # NewType is not a real class; is_subclass returns False gracefully
        assert is_subclass(UserId(1), UserId) is False

        # Final/ClassVar are treated as matching their wrapped type
        assert is_subclass(1, Final[int]) is True
        assert is_subclass(1, ClassVar[int]) is True


class TestGetAttrDescriptors:
    """``get_attr`` with descriptors and properties."""

    def test_get_attr_custom_descriptor(self) -> None:
        class Descriptor:
            def __get__(self, obj: Any, type: type[Any] | None = None) -> int:
                return 42

        class A:
            attr = Descriptor()

        class B(A):
            pass

        result = get_attr(B, "attr")
        assert isinstance(result, Descriptor)

    def test_get_attr_property(self) -> None:
        class A:
            @property
            def prop(self) -> int:
                return 42

        class B(A):
            pass

        result = get_attr(B, "prop")
        assert isinstance(result, property)

    def test_get_attr_classmethod(self) -> None:
        class A:
            @classmethod
            def cls_method(cls) -> None:
                pass

        class B(A):
            pass

        result = get_attr(B, "cls_method")
        assert isinstance(result, classmethod)

    def test_get_attr_staticmethod(self) -> None:
        class A:
            @staticmethod
            def st_method() -> None:
                pass

        class B(A):
            pass

        result = get_attr(B, "st_method")
        assert isinstance(result, staticmethod)

    def test_get_attr_data_descriptor(self) -> None:
        class Desc:
            def __set_name__(self, owner: type[Any], name: str) -> None:
                pass

        class A:
            attr = Desc()

        result = get_attr(A, "attr")
        assert isinstance(result, Desc)


class TestGetAttrProperties:
    """``get_attr`` with property-like constructs."""

    def test_get_attr_cached_property(self) -> None:
        from functools import cached_property

        class A:
            @cached_property
            def cached(self) -> int:
                return 42

        class B(A):
            pass

        result = get_attr(B, "cached")
        assert isinstance(result, cached_property)

    def test_get_attr_property_with_setter(self) -> None:
        class A:
            @property
            def prop(self) -> int:
                return 42

            @prop.setter
            def prop(self, value: int) -> None:
                pass

        result = get_attr(A, "prop")
        assert isinstance(result, property)


class TestGetAttrGetattr:
    """``get_attr`` interaction with ``__getattr__``."""

    def test_get_attr_does_not_trigger_getattr(self) -> None:
        class A:
            def __getattr__(self, name: str) -> str:
                return f"dynamic_{name}"

        class B(A):
            pass

        # get_attr looks at __dict__ entries, so __getattr__ is not triggered
        assert get_attr(B, "dynamic_attr") is None
        assert get_attr(B, "dynamic_attr", "default") == "default"

    def test_get_attr_real_attribute_still_found(self) -> None:
        class A:
            x = 1

            def __getattr__(self, name: str) -> str:
                return f"dynamic_{name}"

        class B(A):
            pass

        assert get_attr(B, "x") == 1


class TestIterInheritanceABC:
    """``iter_inheritance`` with abstract base classes."""

    def test_iter_inheritance_with_abc_custom(self) -> None:
        from collections.abc import Mapping, Sequence

        class MyMap(Mapping):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                return key

            def __iter__(self) -> Iterator[Any]:
                return iter([])

            def __len__(self) -> int:
                return 0

        class MySeq(Sequence):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                return 0

            def __len__(self) -> int:
                return 0

        result = list(iter_inheritance(MyMap, exclude_stdlib=False))
        assert Mapping in result

        result = list(iter_inheritance(MySeq, exclude_stdlib=False))
        assert Sequence in result

    def test_iter_inheritance_abc_exclude(self) -> None:
        from collections.abc import Mapping

        class MyMap(Mapping):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                return key

            def __iter__(self) -> Iterator[Any]:
                return iter([])

            def __len__(self) -> int:
                return 0

        result = list(
            iter_inheritance(
                MyMap,
                exclude=[Mapping],
                exclude_stdlib=False,
            ),
        )
        assert Mapping not in result

    def test_iter_inheritance_abc_include(self) -> None:
        from collections.abc import Mapping

        class MyMap(Mapping):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                return key

            def __iter__(self) -> Iterator[Any]:
                return iter([])

            def __len__(self) -> int:
                return 0

        result = list(
            iter_inheritance(
                MyMap,
                include=[Mapping],
                exclude_stdlib=False,
            ),
        )
        assert result == [Mapping]

    def test_iter_inheritance_multiple_abc(self) -> None:
        from collections.abc import Mapping, MutableMapping

        class MyMap(MutableMapping):  # type: ignore[type-arg]
            def __getitem__(self, key: Any) -> Any:
                return key

            def __setitem__(self, key: Any, value: Any) -> None:
                pass

            def __delitem__(self, key: Any) -> None:
                pass

            def __iter__(self) -> Iterator[Any]:
                return iter([])

            def __len__(self) -> int:
                return 0

        result = list(iter_inheritance(MyMap, exclude_stdlib=False))
        assert MutableMapping in result
        assert Mapping in result


class TestTypingIntrospectionAdvanced:
    """Advanced typing introspection."""

    def test_object_name_generic_nested(self) -> None:
        alias = dict[str, list[int]]
        assert object_name(alias) == "dict"

    def test_object_name_list_of_union(self) -> None:
        alias = list[int | str]
        assert object_name(alias) == "list"

    def test_who_is_union_type(self) -> None:
        result = who_is(int | str)
        assert result == "types.UnionType"

    def test_who_is_generic_alias(self) -> None:
        result = who_is(list[int])
        assert result == "list"

    def test_get_module_literal(self) -> None:
        mod = get_module(Literal[1])
        assert mod is not None
        assert mod.__name__ == "typing"

    def test_get_module_name_literal(self) -> None:
        assert get_module_name(Literal[1]) == "typing"


class TestIsInternalTypingAdvanced:
    """Advanced ``is_internal`` for typing constructs."""

    @pytest.mark.parametrize(
        "obj",
        [
            Any,
            UnionType,
            list[int],
            int | str,
            Literal[1],
            Protocol,
            Generic,
        ],
    )
    def test_internal_true_typing(self, obj: Any) -> None:
        assert is_internal(obj) is True

    def test_internal_false_custom(self) -> None:
        class Custom:
            pass

        assert is_internal(Custom) is False
        assert is_internal(Custom()) is False


class TestWhoIsAdvancedTyping:
    """Advanced ``who_is`` for typing constructs."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (Any, "typing.Any"),
            (UnionType, "types.UnionType"),
            (list[int], "list"),
            (Literal[1], "typing.Literal"),
            (Protocol, "typing.Protocol"),
            (Generic, "typing.Generic"),
            (TypeVar("T"), "tests.test_internals_extended.T"),
            (ParamSpec("P"), "tests.test_internals_extended.P"),
        ],
    )
    def test_who_is_typing(self, obj: Any, expected: str) -> None:
        assert who_is(obj) == expected


class TestObjectNameAdvancedTyping:
    """Advanced ``object_name`` for typing constructs."""

    @pytest.mark.parametrize(
        ("obj", "full", "expected"),
        [
            (Any, True, "typing.Any"),
            (UnionType, True, "types.UnionType"),
            (list[int], True, "list"),
            (Literal[1], True, "typing.Literal"),
            (TypeVar("T"), False, "T"),
            (ParamSpec("P"), False, "P"),
            (Protocol, True, "typing.Protocol"),
            (Generic, True, "typing.Generic"),
        ],
    )
    def test_object_name_typing(
        self,
        obj: Any,
        full: bool,
        expected: str,
    ) -> None:
        name = object_name(obj, full=full)
        assert name == expected or name.endswith(expected)

    def test_object_name_union_pipe(self) -> None:
        alias = int | str
        assert object_name(alias) == "types.UnionType"

    def test_object_name_optional(self) -> None:
        alias = Optional[int]
        assert object_name(alias) == "typing.Optional"


class TestIsCollectionAdvanced:
    """Advanced ``is_collection`` edge cases."""

    def test_collection_duck_typed_full(self) -> None:
        class Duck:
            def __getitem__(self, key: Any) -> Any:
                return key

            def __setitem__(self, key: Any, value: Any) -> None:
                pass

            def __delitem__(self, key: Any) -> None:
                pass

        assert is_collection(Duck()) is True

    def test_collection_partial_duck(self) -> None:
        class PartialDuck:
            def __getitem__(self, key: Any) -> Any:
                return key

        assert is_collection(PartialDuck()) is False

    def test_collection_empty_containers(self) -> None:
        assert is_collection([]) is True
        assert is_collection({}) is True
        assert is_collection(()) is True
        assert is_collection(set()) is False
        assert is_collection("") is False
        assert is_collection(b"") is False


class TestIsMappingAdvanced:
    """Advanced ``is_mapping`` edge cases."""

    def test_mapping_empty(self) -> None:
        assert is_mapping({}) is True
        assert is_mapping(dict()) is True

        class MyDict(dict):
            pass

        assert is_mapping(MyDict()) is True
        assert is_mapping(MyDict) is True

    def test_mapping_non_mapping(self) -> None:
        assert is_mapping([]) is False
        assert is_mapping(()) is False
        assert is_mapping(set()) is False
        assert is_mapping("abc") is False
        assert is_mapping(42) is False
        assert is_mapping(None) is False


class TestIsPrimitiveAdvanced:
    """Advanced ``is_primitive`` edge cases."""

    @pytest.mark.parametrize(
        "obj",
        [
            True,
            False,
            None,
            0,
            0.0,
            0j,
            "",
            b"",
            [],
            (),
            {},
            set(),
            deque(),
            bytearray(),
        ],
    )
    def test_primitive_empty_and_falsy(self, obj: Any) -> None:
        assert is_primitive(obj) is True

    @pytest.mark.parametrize(
        "obj",
        [
            object(),
            lambda: 1,
            type("Custom", (), {}),
            Exception(),
        ],
    )
    def test_primitive_false_custom(self, obj: Any) -> None:
        assert is_primitive(obj) is False


class TestIsIterableAdvanced:
    """Advanced ``is_iterable`` edge cases."""

    def test_iterable_duck(self) -> None:
        class DuckIterable:
            def __iter__(self) -> iter[Any]:
                return iter([])

        assert is_iterable(DuckIterable()) is True

    def test_iterable_generator(self) -> None:
        gen = (x for x in range(3))
        assert is_iterable(gen) is True

    def test_iterable_iterator(self) -> None:
        assert is_iterable(iter([1, 2, 3])) is True

    def test_iterable_string_and_bytes(self) -> None:
        assert is_iterable("abc") is True
        assert is_iterable(b"abc") is True

    def test_iterable_non_iterable(self) -> None:
        assert is_iterable(1) is False
        assert is_iterable(3.14) is False
        assert is_iterable(None) is False
        assert is_iterable(object()) is False


class TestGetMroAdvanced:
    """Advanced ``get_mro`` coverage."""

    def test_get_mro_basic(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B) == (A,)

    def test_get_mro_func_and_glue(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B, func=lambda x: x.__name__) == ("A",)
        assert get_mro(B, func=lambda x: x.__name__, glue=" -> ") == "A"

    def test_get_mro_with_stdlib_exclusion(self) -> None:
        class A(Exception):
            pass

        mro = get_mro(A, exclude_stdlib=False)
        assert Exception in mro

        class A:
            pass

        class B(A):
            pass

        result = get_mro(B, glue=" -> ")
        assert isinstance(result, str)
        assert "A" in result


class TestUniqueAdvanced:
    """Advanced ``unique`` coverage."""

    def test_unique_empty(self) -> None:
        assert list(unique([])) == []

    def test_unique_generator(self) -> None:
        gen = (x for x in [1, 1, 2, 2, 3])
        assert list(unique(gen)) == [1, 2, 3]

    def test_unique_preserve_order(self) -> None:
        data = [3, 1, 4, 1, 5, 9, 2, 6]
        assert list(unique(data)) == [3, 1, 4, 5, 9, 2, 6]

    def test_unique_unhashable_with_key(self) -> None:
        data = [{"a": 1}, {"a": 2}, {"a": 1}]
        result = list(unique(data, key=lambda d: d["a"]))
        assert len(result) == 2
        assert result[0] == {"a": 1}
        assert result[1] == {"a": 2}

    def test_unique_key_with_include_set(self) -> None:
        data = [1, 2, 3, 4]
        result = list(unique(data, key=lambda x: x % 2, include={1}))
        assert result == [1]

    def test_unique_dict_with_key_on_values(self) -> None:
        mapping = {"a": 1, "b": 2, "c": 1}
        result = list(unique(mapping, key=lambda k: mapping[k]))
        assert len(result) == 2

    def test_unique_unhashable_items_with_identity_key(self) -> None:
        data = [[1], [2], [1]]
        result = list(unique(data, key=id))
        assert len(result) == 3


class TestSourceFileAdvanced:
    """Advanced ``source_file`` coverage."""

    def test_source_file_builtin_none(self) -> None:
        assert source_file(1) is None

    def test_source_file_custom_exclude_stdlib(self) -> None:
        class Custom:
            pass

        path = source_file(Custom, exclude_stdlib=False)
        assert path is None or path.endswith(".py")

    def test_source_file_with_template(self) -> None:
        class Custom:
            pass

        path = source_file(Custom, template="file://%s", exclude_stdlib=False)
        assert path is None or path.startswith("file://")
