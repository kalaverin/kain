from asyncio import ensure_future, iscoroutinefunction
from contextlib import suppress
from functools import cached_property, lru_cache, partial, wraps
from inspect import iscoroutine, isfunction, ismethod
from logging import getLogger
from operator import attrgetter
from time import time

from kain.classes import Missing
from kain.internals import (
    Is,
    Who,
    get_attr,
    get_owner,
)

__all__ = 'cache', 'class_property', 'mixed_property', 'pin', 'proxy_to'

Nothing = Missing()
logger = getLogger(__name__)


class PropertyError(Exception): ...


class ContextFaultError(PropertyError): ...


class ReadOnlyError(PropertyError): ...


class AttributeException(PropertyError): ...  # noqa: N818


def cache(limit=None):

    function = partial(lru_cache, maxsize=None, typed=False)

    if isinstance(limit, classmethod | staticmethod):
        msg = f"can't wrap {Who.Is(limit)}, you must use @cache after it"
        raise TypeError(msg)

    for func in isfunction, iscoroutine, ismethod:
        if func(limit):
            return function()(limit)

    if limit is not None and (not isinstance(limit, float | int) or limit <= 0):
        msg = f'limit must be None or positive integer, not {Who.Is(limit)}'
        raise TypeError(msg)

    return function(maxsize=limit) if limit else function()


def extract_wrapped(desc):
    # when it's default instance-method replacer
    if Is.subclass(desc, InsteadProperty):
        return desc.__get__

    # when it's full-featured (cached) property
    if Is.subclass(desc, AbstractProperty):
        return desc.call

    # when it's builtin @property
    if Is.subclass(desc, property):
        return desc.fget

    # when wrapped functions stored in .func
    if Is.subclass(desc, cached_property):
        return desc.func

    msg = (
        f"couldn't extract wrapped function from {Who(desc)}: "
        f"replace it with @property, @cached_property, @{Who(pin)}, "
        f"or other descriptor derived from {Who(AbstractProperty)}"
    )
    raise NotImplementedError(msg)


def parent_call(func):

    @wraps(func)
    def parent_caller(node, *args, **kw):
        try:
            desc = get_attr(
                Is.classOf(node),
                func.__name__,
                exclude_self=True,
                index=func.__name__ not in Is.classOf(node).__dict__,
            )

            return func(node, extract_wrapped(desc)(node, *args, **kw), *args, **kw)

        except RecursionError as e:
            msg = (
                f"{Who(node)}.{func.__name__} call real {Who(func)}, "
                f"couldn't reach parent descriptor; "
                f"maybe {Who(func)} it's mixin of {Who(node)}?"
            )
            raise RecursionError(msg) from e

    return parent_caller


def invokation_context_check(func):

    @wraps(func)
    def context(self, node, *args, **kw):
        if (klass := self.klass) is not None and (
            node is None or klass != Is.Class(node)
        ):
            msg = f'{Who(func)} exception, {self.header_with_context(node)}, {node=}'

            if node is None and not klass:
                msg = f'{msg}; looks like as non-instance invokation'
            raise ContextFaultError(msg)

        return func(self, node, *args, **kw)

    return context


class AbstractProperty:
    @classmethod
    def with_parent(cls, function):
        return cls(parent_call(function))

    def __init__(self, function):
        self.function = function

    @cached_property
    def name(self):
        return self.function.__name__

    @property
    def title(self):
        raise NotImplementedError

    @cached_property
    def header(self):
        try:
            return f'{self.title}({self.function!a})'
        except Exception:  # noqa: BLE001
            return f'{self.title}({Who(self.function)})'

    def header_with_context(self, node):
        raise NotImplementedError


class CustomCallbackMixin:
    @classmethod
    def by(cls, callback):
        if not Is.subclass(cls, Cached):
            cls = Cached  # noqa: PLW0642
        return partial(cls, is_actual=callback)

    expired_by = by

    @classmethod
    def ttl(cls, expire: float):
        if not isinstance(expire, float | int):
            msg = f'expire must be float or int, not {Who.Cast(expire)}'
            raise TypeError(msg)

        if expire <= 0:
            msg = f'expire must be positive number, not {expire!r}'
            raise ValueError(msg)

        def is_actual(self, node, value=Nothing):  # noqa: ARG001
            return (value + expire > time()) if value else time()

        return cls.by(is_actual)


class InsteadProperty(AbstractProperty, CustomCallbackMixin):
    def __init__(self, function):
        if iscoroutinefunction(function):
            msg = (
                f'{Who(function)} is coroutine function, '
                'you must use @pin.native instead of just @pin'
            )
            raise TypeError(msg)
        super().__init__(function)

    @cached_property
    def title(self):
        return f'instance just-replace-descriptor {Who(self, addr=True)}'

    def header_with_context(self, node):
        return (
            f'{self.header} called with '
            f'{("instance", "class")[Is.Class(node)]} '
            f'({Who(node, addr=True)})'
        )

    def __get__(self, node, klass=Nothing):
        if node is None:
            raise ContextFaultError(self.header_with_context(klass))

        with suppress(KeyError):
            return node.__dict__[self.name]

        value = self.function(node)
        node.__dict__[self.name] = value
        return value

    def __delete__(self, node):
        msg = f'{self.header_with_context(node)}: deleter called'
        raise ReadOnlyError(msg)


class BaseProperty(AbstractProperty):

    klass = False
    readonly = False

    @InsteadProperty
    def is_data(self):
        return bool(hasattr(self, '__set__') or hasattr(self, '__delete__'))

    @InsteadProperty
    def title(self):
        mode = 'mixed' if self.klass is None else ('instance', 'class')[self.klass]

        prefix = ('', 'data ')[self.is_data]
        return f'{mode} {prefix}descriptor {Who(self, addr=True)}'.strip()

    def header_with_context(self, node):
        if node is None:
            mode = 'mixed' if self.klass is None else 'undefined'
        else:
            mode = ('instance', 'class')[Is.Class(node)]
        return (
            f'{self.header} with {mode} type called with'
            f'{("instance", "class")[Is.Class(node)]} '
            f'({Who(node, addr=True)})'
        )

    @invokation_context_check
    def get_node(self, node):
        return node

    @invokation_context_check
    def call(self, node):
        try:
            value = self.function(node)
            if not iscoroutinefunction(self.function):
                return value
            return ensure_future(value)

        except AttributeError as e:
            error = AttributeException(str(e).rsplit(':', 1)[-1])

            error.exception = e
            raise error from e

    def __str__(self):
        return f'<{self.header}>'

    def __repr__(self):
        return f'<{self.title}>'

    def __get__(self, instance, klass):
        if instance is None and self.klass is False:
            raise ContextFaultError(self.header_with_context(klass))

        return self.call((instance, klass)[self.klass])


class InheritedClass(BaseProperty):
    """By default class property will be used parent class.
    This class change behavior to last inherited child.
    """

    @invokation_context_check
    def get_node(self, node):
        return node

    @classmethod
    def make_from(cls, parent):
        """Make child-aware class from plain parent-based."""

        name = Who(parent, full=False)
        suffix = f'{"_" if name == name.lower() else ""}inherited'.capitalize()

        result = type(f'{name}{suffix}', (cls, parent), {})
        result.here = parent
        return result


class Cached(BaseProperty, CustomCallbackMixin):

    def __init__(self, function, is_actual=Nothing):
        super().__init__(function)

        if method := getattr(Is.classOf(self), 'is_actual', None):
            if is_actual:
                msg = (
                    f"{Who.Is(self)}.is_actual method ({Who.Cast(method)}) "
                    f"can't override by is_actual kw: {Who.Cast(is_actual)}"
                )
                raise TypeError(msg)
            is_actual = method
        self.is_actual = is_actual

    @invokation_context_check
    def get_cache(self, node):
        name = f'__{("instance", "class")[Is.Class(node)]}_memoized__'

        if hasattr(node, '__dict__'):
            with suppress(KeyError):
                return node.__dict__[name]

        cache = {}
        setattr(node, name, cache)
        return cache

    @invokation_context_check
    def call(self, obj):
        node = self.get_node(obj)
        with suppress(KeyError):

            stored = self.get_cache(node)[self.name]
            if not self.is_actual:
                return stored

            value, stamp = stored
            if self.is_actual(self, node, stamp) is True:
                return value

        return self.__set__(node, super().call(obj))

    @invokation_context_check
    def __set__(self, node, value):
        cache = self.get_cache(node)

        if not self.is_actual:
            cache[self.name] = value
        else:
            cache[self.name] = value, self.is_actual(self, node)
        return value

    @invokation_context_check
    def __delete__(self, node):
        cache = self.get_cache(node)
        with suppress(KeyError):
            del cache[self.name]


class ClassProperty(BaseProperty):
    klass = True

    @invokation_context_check
    def get_node(self, node):
        return get_owner(node, self.name) if Is.Class(node) else node


class MixedProperty(ClassProperty):
    klass = None

    def __get__(self, instance, klass):
        return self.call(instance or klass)


class ClassCachedProperty(ClassProperty, Cached):
    """Class-level cached property that passes the original class as the first
    positional argument and replaces the original data-descriptor."""


class MixedCachedProperty(MixedProperty, Cached):
    """Mixed-level cached property that replaces the original data-descriptor"""


class PreCachedProperty(MixedProperty, Cached):

    @invokation_context_check
    def __set__(self, node, value):
        if not Is.Class(node):
            return value
        return super().__set__(node, value)


class PostCachedProperty(MixedProperty, Cached):

    @invokation_context_check
    def __set__(self, node, value):
        if Is.Class(node):
            return value
        return super().__set__(node, value)


#


class pin(InsteadProperty):  # noqa: N801

    native = Cached
    cls = InheritedClass.make_from(ClassCachedProperty)
    any = InheritedClass.make_from(MixedCachedProperty)
    pre = InheritedClass.make_from(PreCachedProperty)
    post = InheritedClass.make_from(PostCachedProperty)


class class_property(ClassProperty): ...  # noqa: N801


class mixed_property(MixedProperty): ...  # noqa: N801


#


def proxy_to(  # noqa: PLR0915
    *mapping,
    getter=attrgetter,
    default=Nothing,
    pre=None,
    safe=True,
):
    if isinstance(mapping[-1], str):
        bind = pin

    elif mapping[-1] is None:
        bind, mapping = None, mapping[:-1]

    else:
        bind, mapping = mapping[-1], mapping[:-1]

    def class_wraper(cls):  # noqa: PLR0915
        if not Is.Class(cls):
            msg = f"{Who.Is(cls)} isn't a class"
            raise TypeError(msg)

        try:
            fields = cls.__proxy_fields__
        except AttributeError:
            fields = []
            cls.__proxy_fields__ = fields

        pivot, mapping_list = mapping[0], mapping[1:]

        if not mapping_list or (
            len(mapping_list) == 1 and not isinstance(mapping_list[0], str)
        ):
            raise ValueError(f'empty {mapping_list=} for {pivot=}')

        for method in mapping_list:

            if safe and not method.startswith('_') and get_attr(cls, method):
                msg = (
                    f'{Who(cls)} already exists {method!a}: {get_attr(cls, method)}'
                )
                raise TypeError(msg)

            def wrapper(name, node):
                if not isinstance(pivot, str):
                    try:
                        return getattr(pivot, name)
                    except AttributeError as e:
                        msg = (
                            f"{Who(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                            f"{Who(pivot)}.{name}, but last isn't exists"
                        )
                        raise AttributeError(msg) from e

                try:
                    entity = getattr(node, pivot)
                except AttributeError as e:
                    msg = (
                        f"{Who(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                        f"{Who(node)}.{pivot}.{name}, but "
                        f"{Who(node)}.{pivot} isn't exists"
                    )
                    raise AttributeError(msg) from e

                if entity is None:
                    msg = (
                        f'{Who(node)}.{name} {Who.Name(getter)[:4]}-proxied -> '
                        f'{Who(node)}.{pivot}.{name}, but current '
                        f'{Who(node)}.{pivot} is None'
                    )

                    if default is Nothing:
                        raise AttributeError(msg)

                    msg = f'{msg}; return {Who.Is(default)}'
                    logger.warning(msg)
                    result = default

                else:
                    try:
                        result = getter(name)(entity)

                    except (AttributeError, KeyError) as e:
                        msg = (
                            f"{Who(node)}.{name} {Who.Name(getter)[:4]}-proxied -> "
                            f"{Who(node)}.{pivot}.{name}, but isn't exists "
                            f"('{name}' not in {Who(node)}.{pivot}): "
                            f"{Who.Is(entity)}"
                        )

                        if default is Nothing:
                            raise Is.classOf(e)(msg) from e

                        msg = f'{msg}; return {Who.Is(default)}'
                        logger.warning(msg)
                        result = default

                return partial(pre, result) if pre else result

            wrapper.__name__ = method
            wrapper.__qualname__ = f'{pivot}.{method}'

            if bind is None:
                node = cls.__dict__[pivot]
                try:
                    value = node.__dict__[method]
                except KeyError:
                    value = getattr(node, method)
            else:
                wrap = partial(wrapper, method)
                wrap.__name__ = method
                wrap.__qualname__ = f'{pivot}.{method}'
                value = bind(wrap)

            fields.append(method)
            setattr(cls, method, value)
            cls.__proxy_fields__.sort()

        return cls

    return class_wraper
