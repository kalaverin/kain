"""Tests for kain.internals module."""

from __future__ import annotations

import os
import sys
import types
import weakref
from collections import deque
from dataclasses import dataclass
from enum import Enum
from functools import partial, wraps
from typing import Any, Generic, List, Literal, ParamSpec, Protocol, TypeVar

import pytest

from kain.internals import (
    Builtins,
    Collections,
    Primitives,
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
    is_from_primivite,
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
        assert is_from_primivite(1) is True
        assert is_from_primivite("hello") is True
        assert is_from_primivite(None) is True
        assert is_from_primivite(True) is True

    def test_from_primitive_false(self) -> None:
        assert is_from_primivite([1, 2, 3]) is False
        assert is_from_primivite({"a": 1}) is False


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
        assert object_name(List) == "typing.List"
        assert object_name(List[int]) == "typing.List"
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
