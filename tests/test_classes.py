"""Tests for kain.classes module."""

from __future__ import annotations

import copy
import pickle

import pytest

from kain.classes import Missing, Nothing, Singleton


class TestMissing:
    """Tests for Missing sentinel class."""

    def test_bool_is_false(self) -> None:
        """Missing instances should be falsy."""
        missing = Missing()
        assert bool(missing) is False
        assert not missing

    def test_eq_always_false(self) -> None:
        """Missing instances should never compare equal to anything."""
        missing = Missing()
        assert missing != missing
        assert missing != Nothing
        assert missing != None  # noqa: E711
        assert missing != 42
        assert missing != ""
        assert missing != []

    def test_hash_returns_id(self) -> None:
        """__hash__ should return the object's id."""
        missing = Missing()
        assert hash(missing) == id(missing)

    def test_repr_contains_name_and_addr(self) -> None:
        """__repr__ should include class name and memory address."""
        missing = Missing()
        rep = repr(missing)
        assert "Missing" in rep
        assert f"#{id(missing):x}" in rep

    def test_ne_always_true(self) -> None:
        """!= should always be True for Missing instances."""
        missing = Missing()
        assert missing != missing
        assert missing != Nothing
        assert missing != None  # noqa: E711
        assert missing != 0
        assert missing != False  # noqa: E712

    def test_eq_with_self_is_false(self) -> None:
        """A Missing instance should not equal itself."""
        missing = Missing()
        assert (missing == missing) is False

    def test_eq_with_other_missing_is_false(self) -> None:
        """Two Missing instances should not equal each other."""
        m1 = Missing()
        m2 = Missing()
        assert (m1 == m2) is False

    def test_eq_with_none_is_false(self) -> None:
        assert (Missing() == None) is False  # noqa: E711

    def test_eq_with_true_is_false(self) -> None:
        assert (Missing() == True) is False  # noqa: E712

    def test_eq_with_false_is_false(self) -> None:
        assert (Missing() == False) is False  # noqa: E712

    def test_eq_with_int_is_false(self) -> None:
        assert (Missing() == 42) is False

    def test_eq_with_zero_is_false(self) -> None:
        assert (Missing() == 0) is False

    def test_eq_with_float_is_false(self) -> None:
        assert (Missing() == 3.14) is False

    def test_eq_with_string_is_false(self) -> None:
        assert (Missing() == "foo") is False

    def test_eq_with_empty_string_is_false(self) -> None:
        assert (Missing() == "") is False

    def test_eq_with_list_is_false(self) -> None:
        assert (Missing() == [1, 2]) is False

    def test_eq_with_empty_list_is_false(self) -> None:
        assert (Missing() == []) is False

    def test_eq_with_dict_is_false(self) -> None:
        assert (Missing() == {"a": 1}) is False

    def test_eq_with_empty_dict_is_false(self) -> None:
        assert (Missing() == {}) is False

    def test_eq_with_tuple_is_false(self) -> None:
        assert (Missing() == (1, 2)) is False

    def test_eq_with_empty_tuple_is_false(self) -> None:
        assert (Missing() == ()) is False

    def test_eq_with_set_is_false(self) -> None:
        assert (Missing() == {1, 2}) is False

    def test_eq_with_object_is_false(self) -> None:
        assert (Missing() == object()) is False

    def test_ne_with_self_is_true(self) -> None:
        assert (Missing() != Missing()) is True

    def test_ne_with_none_is_true(self) -> None:
        assert (Missing() != None) is True  # noqa: E711

    def test_ne_with_int_is_true(self) -> None:
        assert (Missing() != 42) is True

    def test_hash_consistency(self) -> None:
        m = Missing()
        assert hash(m) == hash(m)

    def test_hash_differs_between_instances(self) -> None:
        m1 = Missing()
        m2 = Missing()
        assert hash(m1) != hash(m2)

    def test_hash_usable_in_set(self) -> None:
        m = Missing()
        s = {m}
        assert m in s

    def test_two_instances_in_set(self) -> None:
        m1 = Missing()
        m2 = Missing()
        s = {m1, m2}
        assert len(s) == 2

    def test_usable_as_dict_key(self) -> None:
        m = Missing()
        d = {m: "value"}
        assert d[m] == "value"

    def test_bool_conversion(self) -> None:
        assert bool(Missing()) is False

    def test_truthiness_in_if(self) -> None:
        m = Missing()
        if m:
            pytest.fail("Missing should be falsy")

    def test_truthiness_in_and(self) -> None:
        m = Missing()
        assert (m and "right") is m

    def test_truthiness_in_or(self) -> None:
        assert (Missing() or "right") == "right"

    def test_type_is_missing(self) -> None:
        assert type(Missing()) is Missing

    def test_isinstance_missing(self) -> None:
        assert isinstance(Missing(), Missing)

    def test_copy_returns_distinct_instance(self) -> None:
        m = Missing()
        c = copy.copy(m)
        assert c is not m
        assert isinstance(c, Missing)

    def test_copy_preserves_behavior(self) -> None:
        c = copy.copy(Missing())
        assert bool(c) is False
        assert c != c

    def test_deepcopy_returns_distinct_instance(self) -> None:
        m = Missing()
        d = copy.deepcopy(m)
        assert d is not m
        assert isinstance(d, Missing)

    def test_deepcopy_preserves_behavior(self) -> None:
        d = copy.deepcopy(Missing())
        assert bool(d) is False
        assert d != d

    @pytest.mark.parametrize("protocol", range(pickle.HIGHEST_PROTOCOL + 1))
    def test_pickle_roundtrip(self, protocol: int) -> None:
        m = Missing()
        data = pickle.dumps(m, protocol=protocol)
        loaded = pickle.loads(data)
        assert isinstance(loaded, Missing)
        assert bool(loaded) is False

    def test_pickle_unpickled_is_distinct(self) -> None:
        m = Missing()
        loaded = pickle.loads(pickle.dumps(m))
        assert loaded is not m

    def test_pickle_preserves_behavior(self) -> None:
        loaded = pickle.loads(pickle.dumps(Missing()))
        assert loaded != loaded
        assert hash(loaded) == id(loaded)

    def test_subclass_can_be_created(self) -> None:
        class MyMissing(Missing):
            pass

        assert issubclass(MyMissing, Missing)

    def test_subclass_instance_is_falsy(self) -> None:
        class MyMissing(Missing):
            pass

        assert bool(MyMissing()) is False

    def test_subclass_instance_never_equal(self) -> None:
        class MyMissing(Missing):
            pass

        mm = MyMissing()
        assert mm != mm
        assert mm != Missing()

    def test_subclass_hash_matches_id(self) -> None:
        class MyMissing(Missing):
            pass

        mm = MyMissing()
        assert hash(mm) == id(mm)

    def test_identity_not_equal_to_equality(self) -> None:
        m = Missing()
        assert m is m
        assert (m == m) is False


class TestNothing:
    """Tests for the Nothing singleton sentinel."""

    def test_is_missing_instance(self) -> None:
        """Nothing should be an instance of Missing."""
        assert isinstance(Nothing, Missing)

    def test_is_falsy(self) -> None:
        """Nothing should be falsy."""
        assert bool(Nothing) is False

    def test_singleton_identity(self) -> None:
        """Nothing should always be the same object."""
        from kain.classes import Nothing as Nothing2

        assert Nothing is Nothing2

    def test_nothing_is_nothing(self) -> None:
        assert Nothing is Nothing

    def test_nothing_type_is_missing(self) -> None:
        assert type(Nothing) is Missing

    def test_nothing_eq_nothing_is_false(self) -> None:
        assert (Nothing == Nothing) is False

    def test_nothing_ne_nothing_is_true(self) -> None:
        assert (Nothing != Nothing) is True

    def test_nothing_eq_missing_instance_is_false(self) -> None:
        assert (Nothing == Missing()) is False

    def test_nothing_ne_missing_instance_is_true(self) -> None:
        assert (Nothing != Missing()) is True

    def test_nothing_hash_matches_id(self) -> None:
        assert hash(Nothing) == id(Nothing)

    def test_nothing_repr_contains_addr(self) -> None:
        rep = repr(Nothing)
        assert f"#{id(Nothing):x}" in rep

    def test_nothing_repr_format(self) -> None:
        rep = repr(Nothing)
        assert rep.startswith("<")
        assert rep.endswith(">")

    def test_nothing_bool_false(self) -> None:
        assert bool(Nothing) is False

    def test_nothing_truthiness_in_if(self) -> None:
        if Nothing:
            pytest.fail("Nothing should be falsy")

    def test_nothing_and_operation(self) -> None:
        assert (Nothing and "x") is Nothing

    def test_nothing_or_operation(self) -> None:
        assert (Nothing or "x") == "x"

    def test_nothing_in_set(self) -> None:
        s = {Nothing}
        assert len(s) == 1
        assert Nothing in s

    def test_nothing_as_dict_key(self) -> None:
        d = {Nothing: 1}
        assert d[Nothing] == 1

    def test_nothing_copy_returns_distinct(self) -> None:
        c = copy.copy(Nothing)
        assert c is not Nothing
        assert isinstance(c, Missing)

    def test_nothing_deepcopy_returns_distinct(self) -> None:
        d = copy.deepcopy(Nothing)
        assert d is not Nothing
        assert isinstance(d, Missing)

    def test_nothing_pickle_roundtrip(self) -> None:
        data = pickle.dumps(Nothing)
        loaded = pickle.loads(data)
        assert isinstance(loaded, Missing)
        assert bool(loaded) is False

    def test_nothing_pickle_preserves_behavior(self) -> None:
        loaded = pickle.loads(pickle.dumps(Nothing))
        assert loaded != loaded

    def test_nothing_not_equal_none(self) -> None:
        assert Nothing != None  # noqa: E711

    def test_nothing_not_equal_zero(self) -> None:
        assert Nothing != 0

    def test_nothing_not_equal_empty_string(self) -> None:
        assert Nothing != ""

    def test_nothing_not_equal_empty_list(self) -> None:
        assert Nothing != []


class TestSingleton:
    """Tests for the Singleton metaclass."""

    def test_returns_same_instance(self) -> None:
        """A class using Singleton should return the same instance."""

        class Single(metaclass=Singleton):
            pass

        a = Single()
        b = Single()
        assert a is b

    def test_instance_created_once(self) -> None:
        """__init__ should only be called once for the singleton."""
        calls: list[tuple[object, ...]] = []

        class Single(metaclass=Singleton):
            def __init__(self, value: object) -> None:
                calls.append((value,))

        s1 = Single(1)
        s2 = Single(2)
        assert s1 is s2
        assert len(calls) == 1
        assert calls[0] == (1,)

    def test_instance_attribute_reset(self) -> None:
        """Resetting the metaclass instance attribute creates a new singleton."""

        class Single(metaclass=Singleton):
            pass

        a = Single()
        Single.instance = Nothing
        b = Single()
        assert a is not b
        c = Single()
        assert b is c

    def test_singleton_with_args(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, a: object, b: object) -> None:
                self.ab = (a, b)

        s1 = Single(1, 2)
        s2 = Single(3, 4)
        assert s1 is s2
        assert s1.ab == (1, 2)

    def test_singleton_with_kwargs(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, x: object = None) -> None:
                self.x = x

        s1 = Single(x="first")
        s2 = Single(x="second")
        assert s1 is s2
        assert s1.x == "first"

    def test_singleton_with_args_and_kwargs(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, a: object, b: object = None) -> None:
                self.val = (a, b)

        s1 = Single(1, b=2)
        s2 = Single(3, b=4)
        assert s1 is s2
        assert s1.val == (1, 2)

    def test_singleton_ignores_subsequent_args(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, val: object) -> None:
                self.val = val

        s1 = Single("first")
        s2 = Single("ignored")
        assert s1.val == "first"

    def test_singleton_ignores_subsequent_kwargs(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, val: object = None) -> None:
                self.val = val

        s1 = Single(val="first")
        s2 = Single(val="ignored")
        assert s1.val == "first"

    def test_singleton_ignores_subsequent_mixed(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, a: object, b: object = None) -> None:
                self.val = (a, b)

        s1 = Single(1, b=2)
        s2 = Single(9, b=99)
        assert s1.val == (1, 2)

    def test_singleton_isinstance_check(self) -> None:
        class Single(metaclass=Singleton):
            pass

        s = Single()
        assert isinstance(s, Single)

    def test_singleton_type_check(self) -> None:
        class Single(metaclass=Singleton):
            pass

        s = Single()
        assert type(s) is Single

    def test_attribute_mutation_shared(self) -> None:
        class Single(metaclass=Singleton):
            pass

        s1 = Single()
        s1.attr = "value"
        s2 = Single()
        assert s2.attr == "value"

    def test_multiple_classes_independent(self) -> None:
        class A(metaclass=Singleton):
            pass

        class B(metaclass=Singleton):
            pass

        a = A()
        b = B()
        assert a is not b
        assert type(a) is A
        assert type(b) is B

    def test_inheritance_from_singleton_class(self) -> None:
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        c1 = Child()
        c2 = Child()
        assert c1 is c2

    def test_subclass_is_singleton(self) -> None:
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        c1 = Child()
        c2 = Child()
        assert c1 is c2

    def test_subclass_independent_of_parent(self) -> None:
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        b = Base()
        c = Child()
        assert b is not c
        assert isinstance(c, Base)

    def test_subclass_isinstance_parent(self) -> None:
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        c = Child()
        assert isinstance(c, Base)

    def test_subclass_isinstance_child(self) -> None:
        class Base(metaclass=Singleton):
            pass

        class Child(Base):
            pass

        c = Child()
        assert isinstance(c, Child)

    def test_singleton_with_no_init(self) -> None:
        class Single(metaclass=Singleton):
            pass

        s1 = Single()
        s2 = Single()
        assert s1 is s2

    def test_singleton_with_default_arg(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, val: object = "default") -> None:
                self.val = val

        s1 = Single()
        s2 = Single("other")
        assert s1 is s2
        assert s1.val == "default"

    def test_singleton_with_varargs(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, *args: object) -> None:
                self.args = args

        s1 = Single(1, 2)
        s2 = Single(3, 4, 5)
        assert s1 is s2
        assert s1.args == (1, 2)

    def test_singleton_with_varkwargs(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, **kw: object) -> None:
                self.kw = kw

        s1 = Single(a=1)
        s2 = Single(b=2)
        assert s1 is s2
        assert s1.kw == {"a": 1}

    def test_singleton_with_new_override(self) -> None:
        class Single(metaclass=Singleton):
            def __new__(cls) -> Single:
                obj = super().__new__(cls)
                obj.created = True
                return obj

        s1 = Single()
        s2 = Single()
        assert s1 is s2
        assert getattr(s1, "created", False)

    def test_singleton_instance_attr_initially_nothing(self) -> None:
        class Single(metaclass=Singleton):
            pass

        assert Single.instance is Nothing

    def test_singleton_class_attr_preserved(self) -> None:
        class Single(metaclass=Singleton):
            cls_attr = "preserved"

        assert Single.cls_attr == "preserved"

    def test_singleton_method_accessible(self) -> None:
        class Single(metaclass=Singleton):
            def method(self) -> str:
                return "ok"

        s = Single()
        assert s.method() == "ok"

    def test_singleton_reset_then_new_instance(self) -> None:
        class Single(metaclass=Singleton):
            pass

        a = Single()
        Single.instance = Nothing
        b = Single()
        assert a is not b

    def test_singleton_str_repr(self) -> None:
        class Single(metaclass=Singleton):
            def __repr__(self) -> str:
                return "<Single>"

        s = Single()
        assert repr(s) == "<Single>"
        assert isinstance(str(s), str)

    def test_singleton_deep_inheritance(self) -> None:
        class A(metaclass=Singleton):
            pass

        class B(A):
            pass

        class C(B):
            pass

        c1 = C()
        c2 = C()
        assert c1 is c2

    def test_singleton_with_classmethod(self) -> None:
        class Single(metaclass=Singleton):
            @classmethod
            def cls_method(cls) -> str:
                return "classmethod"

        s = Single()
        assert Single.cls_method() == "classmethod"
        assert s.cls_method() == "classmethod"

    def test_singleton_with_staticmethod(self) -> None:
        class Single(metaclass=Singleton):
            @staticmethod
            def st_method() -> str:
                return "staticmethod"

        s = Single()
        assert Single.st_method() == "staticmethod"
        assert s.st_method() == "staticmethod"

    def test_singleton_with_property(self) -> None:
        class Single(metaclass=Singleton):
            @property
            def prop(self) -> str:
                return "property"

        s = Single()
        assert s.prop == "property"

    def test_singleton_thread_safety_documented(self) -> None:
        """Singleton checks instance in __call__; CPython GIL protects it."""

        class Single(metaclass=Singleton):
            pass

        assert hasattr(Single, "instance")

    def test_singleton_second_call_no_args_still_returns_first(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, val: object) -> None:
                self.val = val

        s1 = Single("first")
        s2 = Single()
        assert s1 is s2
        assert s1.val == "first"

    def test_singleton_call_with_different_types(self) -> None:
        class Single(metaclass=Singleton):
            def __init__(self, val: object) -> None:
                self.val = val

        s1 = Single([1, 2])
        s2 = Single({"a": 1})
        assert s1 is s2
        assert s1.val == [1, 2]

    def test_singleton_metaclass_sets_instance(self) -> None:
        class Single(metaclass=Singleton):
            pass

        assert hasattr(Single, "instance")
        assert Single.instance is Nothing

    def test_singleton_mro_contains_singleton(self) -> None:
        class Single(metaclass=Singleton):
            pass

        assert Singleton in type(Single).__mro__

    def test_singleton_attribute_set_on_instance_visible(self) -> None:
        class Single(metaclass=Singleton):
            pass

        s = Single()
        s.new_attr = 123
        assert Single().new_attr == 123

    def test_singleton_attribute_set_on_class_visible(self) -> None:
        class Single(metaclass=Singleton):
            pass

        Single.class_attr = 456
        assert Single().class_attr == 456
