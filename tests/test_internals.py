"""Tests for kain.internals module."""

from __future__ import annotations

import os
import sys
import types
import weakref
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from types import GenericAlias, UnionType
from typing import (
    Any,
    Generic,
    Literal,
    Optional,
    ParamSpec,
    Protocol,
    TypeVar,
    Union,
)

import pytest

from kain.internals import (
    Builtins,
    Collections,
    Is,
    Primitives,
    Who,
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
    is_interactive,
    is_internal,
    is_iterable,
    is_mapping,
    is_primitive,
    is_subclass,
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


class TestClassOf:
    def test_class_of_instance(self) -> None:
        assert class_of(42) is int
        assert class_of("hello") is str
        assert class_of([1, 2, 3]) is list

    def test_class_of_class(self) -> None:
        assert class_of(int) is int
        assert class_of(str) is str
        assert class_of(type) is type

    def test_class_of_none(self) -> None:
        assert class_of(None) is type(None)


class TestIsCallable:
    def test_callable_true(self) -> None:
        assert is_callable(lambda: 1) is True
        assert is_callable(print) is True
        assert is_callable(len) is True

    def test_callable_false(self) -> None:
        assert is_callable(1) is False
        assert is_callable("hello") is False
        assert is_callable(None) is False


class TestIsCollection:
    def test_collection_list(self) -> None:
        assert is_collection([1, 2, 3]) is True

    def test_collection_tuple(self) -> None:
        assert is_collection((1, 2, 3)) is True

    def test_collection_dict(self) -> None:
        assert is_collection({"a": 1}) is True

    def test_collection_set(self) -> None:
        # set is Collection but not Sequence and lacks __getitem__
        assert is_collection({1, 2, 3}) is False

    def test_collection_string(self) -> None:
        assert is_collection("abc") is False

    def test_collection_bytes(self) -> None:
        assert is_collection(b"abc") is False

    def test_collection_deque(self) -> None:
        assert is_collection(deque([1, 2, 3])) is True

    def test_collection_custom_mapping(self) -> None:
        class Custom:
            def __getitem__(self, key: Any) -> Any:
                return key

            def __setitem__(self, key: Any, value: Any) -> None:
                pass

            def __delitem__(self, key: Any) -> None:
                pass

        assert is_collection(Custom()) is True


class TestIsIterable:
    def test_iterable_true(self) -> None:
        assert is_iterable([1, 2, 3]) is True
        assert is_iterable("abc") is True
        assert is_iterable({"a": 1}) is True
        assert is_iterable(x for x in range(3)) is True

    def test_iterable_false(self) -> None:
        assert is_iterable(1) is False
        assert is_iterable(None) is False
        assert is_iterable(3.14) is False


class TestIsMapping:
    def test_mapping_dict(self) -> None:
        assert is_mapping({"a": 1}) is True

    def test_mapping_subclass(self) -> None:
        class MyDict(dict):
            pass

        assert is_mapping(MyDict()) is True

    def test_mapping_non_mapping(self) -> None:
        assert is_mapping([]) is False
        assert is_mapping("abc") is False
        assert is_mapping(1) is False


class TestIsPrimitive:
    def test_primitive_true(self) -> None:
        assert is_primitive(1) is True
        assert is_primitive("hello") is True
        assert is_primitive([1, 2, 3]) is True
        assert is_primitive({"a": 1}) is True
        assert is_primitive(None) is True
        assert is_primitive(True) is True
        assert is_primitive(False) is True

    def test_primitive_false(self) -> None:
        class Custom:
            pass

        assert is_primitive(Custom()) is False


class TestIsFromPrimitive:
    def test_from_primitive_true(self) -> None:
        assert is_from_primitive(1) is True
        assert is_from_primitive("hello") is True
        assert is_from_primitive(None) is True
        assert is_from_primitive(True) is True

    def test_from_primitive_false(self) -> None:
        assert is_from_primitive([1, 2, 3]) is False
        assert is_from_primitive({"a": 1}) is False


class TestIsFromBuiltin:
    def test_from_builtin_true(self) -> None:
        assert is_from_builtin([1, 2, 3]) is True
        assert is_from_builtin((1, 2, 3)) is True
        assert is_from_builtin({"a": 1}) is True
        assert is_from_builtin(1) is True
        assert is_from_builtin("hello") is True
        assert is_from_builtin(None) is True

    def test_from_builtin_false(self) -> None:
        class Custom:
            pass

        assert is_from_builtin(Custom()) is False


class TestIsInteractive:
    def test_is_interactive(self) -> None:
        # In test environment this is typically False
        assert is_interactive() is False or is_interactive() is True


class TestConstants:
    def test_collections(self) -> None:
        assert Collections == (deque, dict, list, set, tuple, bytearray)

    def test_primitives(self) -> None:
        assert Primitives == (bool, float, int, str, complex, bytes)

    def test_builtins(self) -> None:
        assert Builtins == Primitives + Collections


class TestIsInternal:
    def test_internal_builtin(self) -> None:
        assert is_internal(1) is True
        assert is_internal(int) is True

    def test_internal_stdlib(self) -> None:
        assert is_internal(os) is True
        assert is_internal(sys) is True

    def test_internal_custom(self) -> None:
        class Custom:
            pass

        assert is_internal(Custom()) is False


class TestIsSubclass:
    def test_subclass_basic(self) -> None:
        assert is_subclass(1, int) is True
        assert is_subclass(1, str) is False

    def test_subclass_none_type(self) -> None:
        assert is_subclass(None, type(None)) is True

    def test_subclass_any(self) -> None:
        assert is_subclass(1, Any) is True

    def test_subclass_union(self) -> None:
        assert is_subclass(1, int | str) is True
        assert is_subclass(1, str | bytes) is False

    def test_subclass_generic(self) -> None:
        assert is_subclass([1, 2], list[int]) is True

    def test_subclass_none_types(self) -> None:
        assert is_subclass(1, None) is False


class TestGetModule:
    def test_get_module_module(self) -> None:
        assert get_module(os) is os

    def test_get_module_object(self) -> None:
        assert get_module(1).__name__ == "builtins"

    def test_get_module_class(self) -> None:
        assert get_module(int).__name__ == "builtins"


class TestGetModuleName:
    def test_get_module_name(self) -> None:
        assert get_module_name(os) == "os"
        assert get_module_name(int) == "builtins"


class TestObjectName:
    def test_object_name_builtin(self) -> None:
        assert object_name(1) == "int"
        assert object_name(int) == "int"
        assert object_name("hello") == "str"
        assert object_name(None) == "NoneType"
        assert object_name(True) == "bool"

    def test_object_name_classes(self) -> None:
        assert object_name(type) == "type"
        assert object_name(object) == "object"

    def test_object_name_module(self) -> None:
        assert object_name(os) == "os"
        assert object_name(sys) == "sys"

    def test_object_name_not_full(self) -> None:
        assert object_name(1, full=False) == "int"
        assert object_name(os, full=False) == "os"

    def test_object_name_builtin_functions(self) -> None:
        assert object_name(print) == "print"
        assert object_name(len) == "len"

    def test_object_name_user_function(self) -> None:
        def my_func() -> None:
            pass

        assert object_name(my_func, full=False) == "my_func"

    def test_object_name_lambda(self) -> None:
        assert object_name(lambda x: x, full=False) == "<lambda>"

    def test_object_name_methods(self) -> None:
        class Foo:
            def bar(self) -> None:
                pass

            @classmethod
            def cls_method(cls) -> None:
                pass

            @staticmethod
            def st_method() -> None:
                pass

        assert object_name(Foo, full=False) == "Foo"
        assert object_name(Foo.bar, full=False) == "bar"
        assert object_name(Foo().bar, full=False) == "bar"
        assert object_name(Foo.cls_method, full=False) == "cls_method"
        assert object_name(Foo().cls_method, full=False) == "cls_method"
        assert object_name(Foo.st_method, full=False) == "st_method"
        assert object_name(Foo().st_method, full=False) == "st_method"
        # full=True retains class qualification
        assert object_name(Foo.bar).endswith(".Foo.bar")

    def test_object_name_property(self) -> None:
        class Foo:
            @property
            def prop(self) -> int:
                return 1

        assert object_name(Foo.prop, full=False) == "prop"
        assert object_name(Foo.prop).endswith(".Foo.prop")

    def test_object_name_async(self) -> None:
        async def async_func() -> None:
            pass

        assert object_name(async_func, full=False) == "async_func"

    def test_object_name_partial(self) -> None:
        def my_func() -> None:
            pass

        assert object_name(partial(my_func)) == "functools.partial"

    def test_object_name_builtin_methods(self) -> None:
        assert object_name([].append) == "list.append"
        assert object_name("".split) == "str.split"
        assert object_name({}.get) == "dict.get"

    def test_object_name_builtin_descriptors(self) -> None:
        assert object_name(list.append) == "list.append"
        assert object_name(object.__init__) == "object.__init__"
        assert object_name([].__add__) == "list.__add__"

    def test_object_name_typing_specials(self) -> None:
        assert object_name(Any) == "typing.Any"

    def test_object_name_enum(self) -> None:
        class Color(Enum):
            RED = 1

        assert object_name(Color, full=False) == "Color"
        assert object_name(Color.RED, full=False) == "Color"

    def test_object_name_named_tuple(self) -> None:
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])
        assert object_name(Point, full=False) == "Point"
        assert object_name(Point(1, 2), full=False) == "Point"

    def test_object_name_dataclass(self) -> None:
        @dataclass
        class Person:
            name: str

        assert object_name(Person, full=False) == "Person"
        assert object_name(Person("bob"), full=False) == "Person"

    def test_object_name_exceptions(self) -> None:
        assert object_name(ValueError) == "ValueError"
        assert object_name(ValueError()) == "ValueError"

    def test_object_name_generator(self) -> None:
        gen = (x for x in range(3))
        assert object_name(gen, full=False) == "<genexpr>"

    def test_object_name_slice(self) -> None:
        assert object_name(slice(1, 2)) == "slice"

    def test_object_name_memoryview(self) -> None:
        assert object_name(memoryview(b"abc")) == "memoryview"

    def test_object_name_weakref(self) -> None:
        class Foo:
            pass

        ref = weakref.ref(Foo())
        assert object_name(ref) == "weakref.ReferenceType"

    def test_object_name_ellipsis(self) -> None:
        assert object_name(...) == "ellipsis"

    def test_object_name_not_implemented(self) -> None:
        assert object_name(NotImplemented) == "NotImplementedType"

    def test_object_name_code(self) -> None:
        code = compile("1+1", "<string>", "eval")
        assert object_name(code) == "code"

    def test_object_name_wrapped_function(self) -> None:
        def original() -> None:
            pass

        @wraps(original)
        def wrapper() -> None:
            pass

        assert object_name(wrapper, full=False) == "original"

    def test_object_name_nested_function(self) -> None:
        def outer() -> None:
            def inner() -> None:
                pass

            return inner

        assert object_name(outer(), full=False) == "inner"

    def test_object_name_typing_constructs(self) -> None:
        assert object_name(TypeVar("T"), full=False) == "T"
        assert object_name(ParamSpec("P"), full=False) == "P"
        assert object_name(Protocol) == "typing.Protocol"
        assert object_name(Generic) == "typing.Generic"
        assert object_name(list) == "list"
        assert object_name(list[int]) == "list"
        assert object_name(Literal[1]) == "typing.Literal"

    def test_object_name_async_generator(self) -> None:
        async def agen() -> None:
            yield 1

        assert object_name(agen, full=False) == "agen"
        assert object_name(agen(), full=False) == "agen"

    def test_object_name_classmethod_staticmethod_descriptors(self) -> None:
        assert object_name(classmethod(lambda x: x)) == "classmethod"
        assert object_name(staticmethod(lambda x: x)) == "staticmethod"


class TestWhoIs:
    def test_who_is_basic(self) -> None:
        assert who_is(1) == "int"
        assert who_is(int) == "int"

    def test_who_is_addr(self) -> None:
        result = who_is(1, addr=True)
        assert result.startswith("int#")

    def test_who_is_not_full(self) -> None:
        assert who_is(1, full=False) == "int"


class TestPrettyModule:
    def test_pretty_module(self) -> None:
        assert pretty_module(1) == "int"


class TestSourceFile:
    def test_source_file_builtin(self) -> None:
        assert source_file(1) is None

    def test_source_file_custom(self) -> None:
        class Custom:
            pass

        result = source_file(Custom)
        assert result is None or result.endswith(".py")


class TestJustValue:
    def test_just_value_builtin(self) -> None:
        assert just_value(1) == "(int)1"

    def test_just_value_class(self) -> None:
        assert just_value(int) == "(int)"


class TestSimpleRepr:
    def test_simple_repr_none(self) -> None:
        assert simple_repr(None) is None

    def test_simple_repr_bool(self) -> None:
        assert simple_repr(True) is True
        assert simple_repr(False) is False

    def test_simple_repr_string(self) -> None:
        assert simple_repr("hello") == "hello"

    def test_simple_repr_number(self) -> None:
        assert simple_repr(42) == "42"
        assert simple_repr(3.14) == "3.14"

    def test_simple_repr_other(self) -> None:
        result = simple_repr([1, 2, 3])
        assert "list" in result


class TestFormatArgsAndKeywords:
    def test_both(self) -> None:
        assert format_args_and_keywords(1, 2, a=3) == "'1', '2', a=3"

    def test_args_only(self) -> None:
        assert format_args_and_keywords(1, 2) == "'1', '2'"

    def test_kwargs_only(self) -> None:
        assert format_args_and_keywords(a=1, b=2) == "a=1, b=2"

    def test_empty(self) -> None:
        assert format_args_and_keywords() == ""


class TestIterInheritance:
    def test_iter_inheritance_basic(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        result = list(iter_inheritance(C))
        assert result == [B, A]

    def test_iter_inheritance_include_self(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B(), exclude_self=False))
        assert result[0].__class__ is B
        assert A in result

    def test_iter_inheritance_reverse(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, reverse=True))
        assert result == [A]

    def test_iter_inheritance_include(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, include=[A]))
        assert result == [A]

    def test_iter_inheritance_exclude(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, exclude=[A]))
        assert result == []


class TestGetMro:
    def test_get_mro_basic(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B) == (A,)

    def test_get_mro_with_func(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B, func=lambda x: x.__name__) == ("A",)

    def test_get_mro_with_glue(self) -> None:
        class A:
            pass

        class B(A):
            pass

        assert get_mro(B, func=lambda x: x.__name__, glue=" -> ") == "A"


class TestGetOwner:
    def test_get_owner(self) -> None:
        class A:
            x = 1

        class B(A):
            pass

        assert get_owner(B, "x") is A

    def test_get_owner_missing(self) -> None:
        class A:
            pass

        assert get_owner(A, "missing") is None


class TestGetAttr:
    def test_get_attr(self) -> None:
        class A:
            x = 1

        class B(A):
            pass

        assert get_attr(B, "x") == 1

    def test_get_attr_default(self) -> None:
        class A:
            pass

        assert get_attr(A, "missing", 999) == 999


class TestIterStack:
    def test_iter_stack(self) -> None:
        result = list(iter_stack(3))
        assert "test_iter_stack" in result


class TestUnique:
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


class TestToAscii:
    def test_to_ascii_str(self) -> None:
        assert to_ascii("hello") == "hello"

    def test_to_ascii_bytes(self) -> None:
        assert to_ascii(b"hello") == "hello"

    def test_to_ascii_invalid(self) -> None:
        with pytest.raises(TypeError):
            to_ascii(123)


class TestToBytes:
    def test_to_bytes_str(self) -> None:
        assert to_bytes("hello") == b"hello"

    def test_to_bytes_bytes(self) -> None:
        assert to_bytes(b"hello") == b"hello"

    def test_to_bytes_invalid(self) -> None:
        with pytest.raises(TypeError):
            to_bytes(123)


class TestIsImportedModule:
    def test_is_imported_module(self) -> None:
        assert is_imported_module("os") is True
        assert is_imported_module("os.path") is True
        assert is_imported_module("missing_module_xyz") is False


class TestIsNamespaceComprehensive:
    """Comprehensive tests for the ``Is`` predicate namespace."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (int, True),
            (str, True),
            (type, True),
            (object, True),
            (42, False),
            ("hello", False),
            (None, False),
            (list[int], False),
            (TypeVar("T"), False),
            (Protocol, True),
            (Generic, True),
            (int | str, False),
        ],
    )
    def test_is_class(self, obj: Any, expected: bool) -> None:
        assert Is.Class(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (lambda: 1, True),
            (print, True),
            (len, True),
            (partial(max, 10), True),
            (object().__str__, True),
            (1, False),
            ("hello", False),
            (None, False),
            ([1, 2], False),
            ({1, 2}, False),
        ],
    )
    def test_is_callable(self, obj: Any, expected: bool) -> None:
        assert Is.callable(obj) is expected

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
            (None, False),
            (42, False),
            (set(), False),
        ],
    )
    def test_is_collection(self, obj: Any, expected: bool) -> None:
        assert Is.collection(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ({"a": 1}, True),
            (dict, True),
            (None, False),
            ([1, 2], False),
            ("abc", False),
            (42, False),
            ((), False),
        ],
    )
    def test_is_mapping(self, obj: Any, expected: bool) -> None:
        assert Is.mapping(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (os, True),
            (sys, True),
            (types, True),
            (os.path, True),
            (42, False),
            ("hello", False),
            (int, False),
            (None, False),
        ],
    )
    def test_is_module(self, obj: Any, expected: bool) -> None:
        assert Is.module(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (True, True),
            (False, True),
            (None, True),
            (1, True),
            (1.5, True),
            (1j, True),
            ("hello", True),
            (b"hello", True),
            ([1, 2], True),
            ((1, 2), True),
            ({"a": 1}, True),
            ({1, 2}, True),
            (deque([1]), True),
            (bytearray(b"x"), True),
            (object(), False),
            (lambda: 1, False),
        ],
    )
    def test_is_primitive(self, obj: Any, expected: bool) -> None:
        assert Is.primitive(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            (1, int, True),
            (1, str, False),
            (None, type(None), True),
            (None, object, True),
            (1, Any, True),
            (1, int | str, True),
            (1, str | bytes, False),
            ([], list[int], True),
            ({}, dict[str, int], True),
            (1, None, False),
            ("x", str | int | bytes, True),
        ],
    )
    def test_is_subclass(self, obj: Any, types: Any, expected: bool) -> None:
        assert Is.subclass(obj, types) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (1, True),
            (int, True),
            (print, True),
            (os, True),
            (sys, True),
            (Any, True),
            (object(), True),
            (type("Custom", (), {}), False),
            (lambda: 1, False),
        ],
    )
    def test_is_internal(self, obj: Any, expected: bool) -> None:
        assert Is.internal(obj) is expected

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ([1, 2], True),
            ("abc", True),
            ({"a": 1}, True),
            ((x for x in range(3)), True),
            (iter([1, 2]), True),
            (1, False),
            (None, False),
            (object(), False),
            (42.0, False),
        ],
    )
    def test_is_iterable(self, obj: Any, expected: bool) -> None:
        assert Is.iterable(obj) is expected

    def test_is_awaitable(self) -> None:
        async def coro() -> None:
            pass

        assert Is.awaitable(coro()) is True
        assert Is.awaitable(1) is False

    def test_is_function(self) -> None:
        def f() -> None:
            pass

        assert Is.function(f) is True
        assert Is.function(lambda: 1) is True
        assert Is.function(1) is False

    def test_is_method(self) -> None:
        class Foo:
            def bar(self) -> None:
                pass

        assert Is.method(Foo().bar) is True
        assert Is.method(Foo.bar) is False
        assert Is.method(1) is False

    def test_is_builtin(self) -> None:
        assert Is.builtin(print) is True
        assert Is.builtin(len) is True
        assert Is.builtin(1) is False

    def test_is_coroutine(self) -> None:
        async def coro() -> None:
            pass

        assert Is.coroutine(coro()) is True
        assert Is.coroutine(1) is False

    def test_is_builtin_predicate(self) -> None:
        assert Is.Builtin([1]) is True
        assert Is.Builtin(object()) is False

    def test_is_primitive_predicate(self) -> None:
        assert Is.Primitive(1) is True
        assert Is.Primitive([1]) is False
        assert Is.Primitive(None) is True

    def test_is_tty_exists(self) -> None:
        assert isinstance(Is.tty, bool)

    def test_is_class_of(self) -> None:
        assert Is.classOf(42) is int
        assert Is.classOf(int) is int
        assert Is.classOf(None) is type(None)

    def test_is_imported(self) -> None:
        assert Is.imported("os") is True
        assert Is.imported("sys") is True
        assert Is.imported("nonexistent_module_12345") is False


class TestWhoNamespaceComprehensive:
    """Comprehensive tests for the ``Who`` introspection namespace."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (1, "int"),
            ("hello", "str"),
            (None, "NoneType"),
            (int, "int"),
        ],
    )
    def test_who_is_basic(self, obj: Any, expected: str) -> None:
        assert Who.Is(obj) == expected

    def test_who_is_addr(self) -> None:
        result = Who.Addr(1)
        assert result.startswith("int#")
        hex_part = result.split("#")[1]
        assert int(hex_part, 16) == id(1)

    def test_who_is_name(self) -> None:
        assert Who.Name(1) == "int"
        assert Who.Name(str) == "str"

    def test_who_module(self) -> None:
        assert Who.Module(int) == "int"

    def test_who_args(self) -> None:
        assert Who.Args(1, 2, a=3) == "'1', '2', a=3"

    def test_who_cast(self) -> None:
        assert Who.Cast(42) == "(int)42"
        assert Who.Cast(int) == "(int)"

    def test_who_inheritance(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = Who.Inheritance(B)
        assert isinstance(result, tuple)
        assert A in result

    def test_who_file(self) -> None:
        assert Who.File(1) is None


class TestWhoIsEdgeCases:
    """Edge cases for ``who_is`` and ``object_name``."""

    @pytest.mark.parametrize(
        ("obj", "full", "expected"),
        [
            (lambda x: x, False, "<lambda>"),
            (object().__str__, False, "__str__"),
            (weakref.ref(type("X", (), {})()), True, "weakref.ReferenceType"),
            (..., True, "ellipsis"),
            (NotImplemented, True, "NotImplementedType"),
            (slice(1, 2), True, "slice"),
            (memoryview(b"abc"), True, "memoryview"),
        ],
    )
    def test_who_is_specials(
        self,
        obj: Any,
        full: bool,
        expected: str,
    ) -> None:
        result = who_is(obj, full=full)
        assert result == expected or result.endswith(expected)

    def test_who_is_method_bound(self) -> None:
        class Foo:
            def bar(self) -> None:
                pass

        bound = Foo().bar
        name = who_is(bound, full=False)
        assert name == "bar"

    def test_who_is_nested_function(self) -> None:
        def outer() -> Any:
            def inner() -> None:
                pass

            return inner

        assert who_is(outer(), full=False) == "inner"

    def test_who_is_module(self) -> None:
        assert who_is(os) == "os"
        assert who_is(sys) == "sys"

    def test_who_is_typing_any(self) -> None:
        assert who_is(Any) == "typing.Any"

    def test_who_is_caching(self) -> None:
        class Dummy:
            pass

        d = Dummy()
        name1 = who_is(d)
        name2 = who_is(d)
        assert name1 == name2
        assert d.__dict__.get("__name_full__") == name1

    def test_who_is_addr_format(self) -> None:
        result = who_is("x", addr=True)
        assert "#" in result
        hex_part = result.split("#")[1]
        assert int(hex_part, 16) == id("x")


class TestObjectNameEdgeCases:
    """Additional ``object_name`` edge cases."""

    @pytest.mark.parametrize(
        ("obj", "full", "expected"),
        [
            (classmethod(lambda x: x), True, "classmethod"),
            (staticmethod(lambda x: x), True, "staticmethod"),
            (property(lambda self: 1), True, "<lambda>"),
        ],
    )
    def test_object_name_descriptors(
        self,
        obj: Any,
        full: bool,
        expected: str,
    ) -> None:
        result = object_name(obj, full=full)
        assert result == expected or result.endswith(expected)

    def test_object_name_generator_expression(self) -> None:
        gen = (x for x in range(3))
        assert object_name(gen, full=False) == "<genexpr>"

    def test_object_name_async_generator(self) -> None:
        async def agen() -> Any:
            yield 1

        assert object_name(agen, full=False) == "agen"
        assert object_name(agen(), full=False) == "agen"

    def test_object_name_partial(self) -> None:
        def my_func() -> None:
            pass

        p = partial(my_func)
        assert object_name(p) == "functools.partial"

    def test_object_name_wrapped(self) -> None:
        def original() -> None:
            pass

        @wraps(original)
        def wrapper() -> None:
            pass

        assert object_name(wrapper, full=False) == "original"

    def test_object_name_typevar(self) -> None:
        assert object_name(TypeVar("T"), full=False) == "T"
        assert object_name(ParamSpec("P"), full=False) == "P"

    def test_object_name_protocol(self) -> None:
        class MyProto(Protocol):
            pass

        result = object_name(MyProto, full=False)
        assert result == "MyProto"


class TestClassOfEdgeCases:
    """Extended ``class_of`` coverage."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            (42, int),
            (int, int),
            (None, type(None)),
            (list[int], GenericAlias),
            (int | str, UnionType),
            (..., type(...)),
            (TypeVar("T"), TypeVar),
            (Literal[1], type(Literal[1])),
            (NotImplemented, type(NotImplemented)),
        ],
    )
    def test_class_of(self, obj: Any, expected: type[Any]) -> None:
        assert class_of(obj) is expected


class TestUniqueEdgeCases:
    """Extended ``unique`` coverage."""

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

    def test_unique_mapping_preserve_order(self) -> None:
        mapping = {"z": 1, "a": 2}
        result = list(unique(mapping))
        keys = [k for k, v in result]
        assert keys == ["z", "a"]

    def test_unique_include_tuple(self) -> None:
        assert list(unique([1, 2, 3, 4], include=(1, 2))) == [1, 2]

    def test_unique_exclude_tuple(self) -> None:
        assert list(unique([1, 2, 3, 4], exclude=(2,))) == [1, 3, 4]

    def test_unique_key_with_include_set(self) -> None:
        data = [1, 2, 3, 4]
        result = list(unique(data, key=lambda x: x % 2, include={1}))
        assert result == [1]

    def test_unique_string_iterable(self) -> None:
        assert list(unique("abac")) == ["a", "b", "c"]

    def test_unique_tuple_iterable(self) -> None:
        assert list(unique((1, 2, 1, 3))) == [1, 2, 3]

    def test_unique_dict_with_key_on_values(self) -> None:
        mapping = {"a": 1, "b": 2, "c": 1}
        result = list(unique(mapping, key=lambda k: mapping[k]))
        assert len(result) == 2

    def test_unique_unhashable_items_with_identity_key(self) -> None:
        data = [[1], [2], [1]]
        result = list(unique(data, key=id))
        assert len(result) == 3

    def test_unique_with_none_key(self) -> None:
        data = [1, 2, 1]
        result = list(unique(data, key=None))
        assert result == [1, 2]


class TestIterInheritanceEdgeCases:
    """Extended ``iter_inheritance`` coverage."""

    def test_iter_inheritance_multiple(self) -> None:
        class A:
            pass

        class B:
            pass

        class C(A, B):
            pass

        result = list(iter_inheritance(C))
        assert A in result
        assert B in result

    def test_iter_inheritance_diamond(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(A):
            pass

        class D(B, C):
            pass

        result = list(iter_inheritance(D))
        assert result == [B, C, A]

    def test_iter_inheritance_stop_classes(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        result = list(iter_inheritance(C, exclude=[A]))
        assert result == [B]

    def test_iter_inheritance_include_callable(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(
            iter_inheritance(B, include=lambda cls: cls.__name__ == "A"),
        )
        assert result == [A]

    def test_iter_inheritance_exclude_callable(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B, exclude=lambda cls: cls is A))
        assert result == []

    def test_iter_inheritance_instance(self) -> None:
        class A:
            pass

        class B(A):
            pass

        result = list(iter_inheritance(B(), exclude_self=False))
        assert result[0].__class__ is B

    def test_iter_inheritance_reverse_multiple(self) -> None:
        class A:
            pass

        class B(A):
            pass

        class C(B):
            pass

        result = list(iter_inheritance(C, reverse=True))
        assert result == [A, B]

    def test_iter_inheritance_with_abc(self) -> None:
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

    def test_iter_inheritance_no_stdlib(self) -> None:
        class MyException(Exception):
            pass

        result = list(iter_inheritance(MyException, exclude_stdlib=True))
        assert Exception not in result

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


class TestGetOwnerEdgeCases:
    """Extended ``get_owner`` coverage."""

    def test_get_owner_nested_class_attribute(self) -> None:
        class Outer:
            x = 1

            class Inner:
                pass

        assert get_owner(Outer.Inner, "x") is None

    def test_get_owner_multiple_inheritance(self) -> None:
        class A:
            x = 1

        class B:
            y = 2

        class C(A, B):
            pass

        assert get_owner(C, "x") is A
        assert get_owner(C, "y") is B

    def test_get_owner_method(self) -> None:
        class A:
            def method(self) -> None:
                pass

        class B(A):
            pass

        assert get_owner(B, "method") is A

    def test_get_owner_with_index(self) -> None:
        class A:
            x = 1

        class B(A):
            x = 2

        class C(B):
            pass

        assert get_owner(C, "x") is B
        assert get_owner(C, "x", index=1) is A
        assert get_owner(C, "x", index=2) is None

    def test_get_owner_property(self) -> None:
        class A:
            @property
            def prop(self) -> int:
                return 1

        class B(A):
            pass

        assert get_owner(B, "prop") is A

    def test_get_owner_classmethod(self) -> None:
        class A:
            @classmethod
            def cls_method(cls) -> None:
                pass

        class B(A):
            pass

        assert get_owner(B, "cls_method") is A

    def test_get_owner_staticmethod(self) -> None:
        class A:
            @staticmethod
            def st_method() -> None:
                pass

        class B(A):
            pass

        assert get_owner(B, "st_method") is A


class TestGetAttrEdgeCases:
    """Extended ``get_attr`` coverage."""

    def test_get_attr_descriptor(self) -> None:
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

    def test_get_attr_default_various(self) -> None:
        class A:
            pass

        assert get_attr(A, "missing") is None
        assert get_attr(A, "missing", "default") == "default"
        assert get_attr(A, "missing", default=[]) == []
        assert get_attr(A, "missing", default=0) == 0

    def test_get_attr_with_index(self) -> None:
        class A:
            x = 1

        class B(A):
            x = 2

        class C(B):
            pass

        assert get_attr(C, "x") == 2
        assert get_attr(C, "x", index=1) == 1
        assert get_attr(C, "x", index=2, default="nope") == "nope"

        class A:
            pass

        result = get_attr(A, "__init__", default="nope")
        assert result == "nope"

    def test_get_attr_data_descriptor(self) -> None:
        class Desc:
            def __set_name__(self, owner: type[Any], name: str) -> None:
                pass

        class A:
            attr = Desc()

        result = get_attr(A, "attr")
        assert isinstance(result, Desc)


class TestToAsciiEdgeCases:
    """Extended ``to_ascii`` coverage."""

    @pytest.mark.parametrize(
        ("inp", "charset", "expected"),
        [
            ("hello", None, "hello"),
            (b"hello", None, "hello"),
            ("café", "utf-8", "café"),
            ("café".encode(), "utf-8", "café"),
            ("naïve", "latin-1", "naïve"),
            ("naïve".encode("latin-1"), "latin-1", "naïve"),
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
        "inp",
        [
            123,
            3.14,
            None,
            [1, 2],
            {"a": 1},
            object(),
        ],
    )
    def test_to_ascii_invalid(self, inp: Any) -> None:
        with pytest.raises(TypeError):
            to_ascii(inp)


class TestToBytesEdgeCases:
    """Extended ``to_bytes`` coverage."""

    @pytest.mark.parametrize(
        ("inp", "charset", "expected"),
        [
            ("hello", None, b"hello"),
            (b"hello", None, b"hello"),
            ("café", "utf-8", "café".encode()),
            ("naïve", "latin-1", "naïve".encode("latin-1")),
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

    @pytest.mark.parametrize(
        "inp",
        [
            123,
            3.14,
            None,
            [1, 2],
            {"a": 1},
            object(),
        ],
    )
    def test_to_bytes_invalid(self, inp: Any) -> None:
        with pytest.raises(TypeError):
            to_bytes(inp)


class TestPrettyModuleEdgeCases:
    """Extended ``pretty_module`` coverage."""

    def test_pretty_module_builtin(self) -> None:
        assert pretty_module(1) == "int"

    def test_pretty_module_stdlib_function(self) -> None:
        result = pretty_module(os.path.join)
        assert result in ("os.path", "posixpath", "ntpath")

    def test_pretty_module_typing(self) -> None:
        assert pretty_module(Any) == "typing"

    def test_pretty_module_custom_class(self) -> None:
        class Foo:
            pass

        result = pretty_module(Foo)
        assert "test_internals" in result


class TestSourceFileEdgeCases:
    """Extended ``source_file`` coverage."""

    def test_source_file_builtin(self) -> None:
        assert source_file(1) is None

    def test_source_file_custom(self) -> None:
        class Custom:
            pass

        result = source_file(Custom, exclude_stdlib=False)
        assert result is None or result.endswith(".py")

    def test_source_file_template(self) -> None:
        class Custom:
            pass

        result = source_file(
            Custom,
            template="file://%s",
            exclude_stdlib=False,
        )
        assert result is None or result.startswith("file://")


class TestJustValueEdgeCases:
    """Extended ``just_value`` coverage."""

    def test_just_value_instance(self) -> None:
        assert just_value(42) == "(int)42"

    def test_just_value_class(self) -> None:
        assert just_value(int) == "(int)"

    def test_just_value_addr(self) -> None:
        s = just_value("hello", addr=True)
        assert "(str#" in s


class TestSimpleReprEdgeCases:
    """Extended ``simple_repr`` coverage."""

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


class TestFormatArgsAndKeywordsEdgeCases:
    """Extended ``format_args_and_keywords`` coverage."""

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

    def test_format_single_arg(self) -> None:
        assert format_args_and_keywords(42) == "'42'"

    def test_format_single_kwarg(self) -> None:
        assert format_args_and_keywords(flag=True) == "flag=True"


class TestIterStackEdgeCases:
    """Extended ``iter_stack`` coverage."""

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

    def test_iter_stack_offset(self) -> None:
        result = list(iter_stack(3, offset=1))
        assert isinstance(result, list)


class TestGetModuleEdgeCases:
    """Extended ``get_module`` and ``get_module_name`` coverage."""

    def test_get_module_typing(self) -> None:
        assert get_module(Any).__name__ == "typing"
        assert get_module(UnionType).__name__ == "types"

    def test_get_module_name_typing(self) -> None:
        assert get_module_name(Any) == "typing"
        assert get_module_name(UnionType) == "types"

    def test_get_module_builtin(self) -> None:
        assert get_module(1).__name__ == "builtins"
        assert get_module_name(1) == "builtins"

    def test_get_module_none(self) -> None:
        class NoModule:
            pass

        instance = NoModule()
        instance.__module__ = None  # type: ignore[attr-defined]
        mod = get_module(instance)
        assert mod is None or mod.__name__ is not None


class TestIsSubclassEdgeCases:
    """Extended ``is_subclass`` coverage."""

    @pytest.mark.parametrize(
        ("obj", "types", "expected"),
        [
            (1, int, True),
            (1, str, False),
            (None, type(None), True),
            (1, Any, True),
            (1, int | str, True),
            (1, str | bytes, False),
            ([], list[int], True),
            ({}, dict[str, int], True),
            (1, None, False),
            (None, object, True),
            ("x", str | int | bytes, True),
        ],
    )
    def test_subclass_basic(
        self,
        obj: Any,
        types: Any,
        expected: bool,
    ) -> None:
        assert is_subclass(obj, types) is expected

    def test_subclass_with_typing_union(self) -> None:
        assert is_subclass(1, Union[int, str]) is True
        assert is_subclass(1, Union[str, bytes]) is False

    def test_subclass_optional(self) -> None:
        assert is_subclass(1, Optional[int]) is True
        assert is_subclass(None, Optional[int]) is True

    def test_subclass_generic_alias(self) -> None:
        assert is_subclass([1, 2], list[int]) is True
        assert is_subclass([1, 2], list[str]) is True
        assert is_subclass({"a": 1}, dict[str, int]) is True

        T = TypeVar("T")
        assert is_subclass(1, T) is False


class TestIsImportedModuleEdgeCases:
    """Extended ``is_imported_module`` coverage."""

    @pytest.mark.parametrize(
        "name",
        [
            "os",
            "os.path",
            "sys",
            "collections.abc",
            "typing",
            "json",
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
        assert is_imported_module("os.path.basename") is True


class TestIsInteractiveEdgeCases:
    """Extended ``is_interactive`` coverage."""

    def test_is_interactive_returns_bool(self) -> None:
        assert isinstance(is_interactive(), bool)

    def test_is_tty_in_is_namespace(self) -> None:
        assert isinstance(Is.tty, bool)
