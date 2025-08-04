import sys
from collections import deque, namedtuple
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from contextlib import suppress
from functools import cache, partial
from inspect import (
    getmodule,
    getsourcefile,
    isawaitable,
    isbuiltin,
    isclass,
    iscoroutine,
    isfunction,
    ismethod,
    ismodule,
    stack,
)
from itertools import filterfalse
from operator import itemgetter, methodcaller
from pathlib import Path
from platform import architecture
from re import sub
from sys import modules, stderr, stdin, stdout
from sysconfig import get_paths
from types import FunctionType, LambdaType, UnionType
from typing import Any, GenericAlias, get_args, get_origin

Collections = deque, dict, list, set, tuple, bytearray
Primitives = bool, float, int, str, complex, bytes
Builtins = Primitives + Collections

WinNT = 'windows' in architecture()[1].lower()

__all__ = (
    'Is',
    'Who',
    'get_attr',
    'get_owner',
    'iter_inheritance',
    'iter_stack',
    'to_ascii',
    'to_bytes',
    'unique',
)


def class_of(obj):
    return obj if isclass(obj) else type(obj)


def is_callable(obj):
    return isinstance(obj, Callable) or callable(obj)


def is_collection(obj):
    return (
        (
            isinstance(obj, Collection)
            and isinstance(obj, Sequence)
            and not isinstance(obj, bytes | str)
        )
        or is_mapping(obj)
        or all(map(partial(hasattr, obj), ('__getitem__', '__setitem__', '__delitem__')))
    )


def is_iterable(obj):
    return isinstance(obj, Iterable) or hasattr(obj, '__iter__')


def is_mapping(obj):
    return isinstance(obj, Mapping) or issubclass(class_of(obj), dict)


def is_primitive(obj):
    return obj is True or obj is False or obj is None or type(obj) in Builtins


def is_from_primivite(obj):
    return bool(obj is None or isinstance(obj, Primitives))


def is_from_builtin(obj):
    return bool(isinstance(obj, Collections) or is_from_primivite(obj))


def is_tty():
    if not getattr(sys, 'frozen', False):  # nuitka compiler checks this
        return all(map(methodcaller('isatty'), (stderr, stdin, stdout)))


@cache
def _get_module_path_type(full):
    dirs = get_paths()

    path = str(full)
    if WinNT:
        path = path.lower()

    for scheme, reason in (
        ('stdlib', True),
        ('purelib', False),
        ('platlib', False),
        ('platstdlib', True),
    ):
        subdir = dirs[scheme]
        if WinNT:
            subdir = subdir.lower()

        if path.startswith(subdir):
            return reason, str(full)[len(subdir) + 1 :]

    subdir = str(Path(__file__).parent.parent)

    if WinNT:
        subdir = subdir.lower()

    if path.startswith(subdir):
        return False, str(full)[len(subdir) + 1 :]

    return None, str(full)


def is_internal(x):
    if isbuiltin(x) or isbuiltin(class_of(x)):
        return True

    if module := get_module(x):

        if module.__name__ == 'builtins':
            return True

        is_stdlib = _get_module_path_type(module.__file__)[0]
        if is_stdlib is not None:
            return is_stdlib

    return False


def is_subclass(obj, types):  # noqa: PLR0911
    if types is None:
        return False

    if types in (Any, obj, object):
        return True

    if obj is None and types is class_of(None):
        return True

    cls = class_of(obj)

    if origin := get_origin(types):
        if cls is origin:
            return True

        args = get_args(types)

        if args and (Any in args or cls in args):
            # Any | None
            return True

        if origin is UnionType and (issubclass(cls, types) or (args and cls in args)):
            return True

        if class_of(types) is GenericAlias:
            # dict[str, str])
            return issubclass(cls, origin)

    return issubclass(cls, types)


def get_module(x):
    if ismodule(x):
        return x

    if (module := getmodule(x)) or (module := getmodule(class_of(x))):
        return module


def get_module_name(x):
    if module := get_module(x):
        with suppress(AttributeError):
            return module.__spec__.name


def object_name(obj, full=True):
    def post(x):
        return sub(r'^([\?\.]+)', '', sub('^(__main__|__builtin__|builtins)', '', x))

    def get_module_from(x):
        return getattr(x, '__module__', get_module_name(x)) or '?'

    def get_object_name(x):
        if obj is Any:
            return 'typing.Any' if full else 'Any'

        name = getattr(x, '__qualname__', x.__name__)
        module = get_module_from(x)

        if not name.startswith(module):
            name = f'{module}.{name}'
        return name

    def main(obj):
        if ismodule(obj):
            return get_module_name(obj)

        for itis in iscoroutine, isfunction, ismethod:
            if itis(obj):
                name = get_object_name(obj)
                with suppress(AttributeError):
                    name = f'{object_name(obj.im_self or obj.im_class)}.{post(name)}'
                return name

        cls = class_of(obj)
        if cls is property:
            return get_object_name(obj.fget)

        return get_object_name(cls)

    name = post(main(obj))
    return name if full else name.rsplit('.', 1)[-1]


def pretty_module(obj):
    return Who(obj).rsplit('.', 1)[0]


def source_file(something, template=None, **kw):
    kw.setdefault('exclude_self', False)
    kw.setdefault('exclude_stdlib', False)

    for obj in iter_inheritance(class_of(something), **kw):
        try:
            if path := getsourcefile(obj):
                return (template % path) if template else str(path)
        except TypeError:  # noqa: PERF203
            ...


def just_value(obj, /, **kw):
    kw.setdefault('addr', False)

    name = Who(obj, **kw)
    if isclass(obj):
        return f'({name})'
    return f'({name}){obj}'


def who_is(obj, /, **kw):  # noqa: N802
    kw.setdefault('addr', True)
    return just_value(obj, **kw)


@cache
def is_imported_module(name):

    with suppress(KeyError):
        return bool(modules[name])

    chunks = name.split('.')
    return (
        sum('.'.join(chunks[: no + 1]) in modules for no in range(len(chunks))) >= 2
    )  # noqa: PLR2004


def get_mro(obj, /, **kw):

    func = kw.pop('func', None)
    glue = kw.pop('glue', None)

    result = iter_inheritance(obj, **kw)

    if func:
        result = tuple(map(func, result))

    if glue:
        result = glue.join(result)

    return result if (func or glue) else tuple(result)


#


def simple_repr(x):
    if (x is None or x is True or x is False) or isinstance(x, str):
        return x

    if isinstance(x, int | float):
        return repr(x)

    return Who.Cast(x)


def format_args_and_keywords(*args, **kw):
    def format_args(x):
        return repr(tuple(map(simple_repr, x)))[1:-1].rstrip(',')

    def format_kwargs(x):
        return ', '.join(f'{k}={simple_repr(v)}' for k, v in x.items())

    if args and kw:
        return f'{format_args(args)}, {format_kwargs(kw)}'

    if args:
        return format_args(args)

    if kw:
        return format_kwargs(kw)

    return ''


# public interface, Is/Who


def Who(obj, /, full=True, addr=False):  # noqa: N802
    key = '__name_full__' if full else '__name_short__'

    def get_name():
        try:
            store = obj.__dict__
            with suppress(KeyError):
                return store[key]
        except AttributeError:
            store = None

        name = object_name(obj, full=full)
        if store is not None:
            with suppress(AttributeError, TypeError):
                setattr(obj, key, name)
        return name

    name = get_name()
    if addr:
        name = f'{name}#{id(obj):x}'
    return name


Who.Args = format_args_and_keywords
Who.Cast = just_value
Who.File = source_file
Who.Inheritance = get_mro
Who.Is = who_is
Who.Module = pretty_module
Who.Name = partial(Who, full=False)

#

Is = namedtuple(
    'Is',
    (
        'Builtin '
        'Class '
        'Primivite '
        'TTY '
        'awaitable '
        'builtin '
        'callable '
        'classOf '
        'collection '
        'coroutine '
        'function '
        'imported '
        'internal '
        'iterable '
        'mapping '
        'method '
        'module '
        'primitive '
        'subclass'
    ),
)(
    is_from_builtin,
    isclass,
    is_from_primivite,
    is_tty,
    isawaitable,
    isbuiltin,
    is_callable,
    class_of,
    is_collection,
    iscoroutine,
    isfunction,
    is_imported_module,
    is_internal,
    is_iterable,
    is_mapping,
    ismethod,
    ismodule,
    is_primitive,
    is_subclass,
)


# public functions


def iter_stack(*args, **kw):
    result = stack()[kw.pop('offset', 0) :]
    yield from (map(itemgetter(*args), result) if args else result)


def iter_inheritance(  # noqa: PLR0913
    obj,
    include=None,
    exclude=None,
    exclude_self=True,
    exclude_stdlib=True,
    reverse=False,
):
    order = class_of(obj).__mro__[:-1]

    if not exclude_self:
        order = unique((obj, *order), key=id)
    else:
        order = unique(filter(lambda x: x is not obj, order), key=id)

    if reverse:
        order = reversed(list(order))

    if include:
        if isinstance(include, FunctionType | LambdaType):
            order = filter(include, order)
        else:
            if not is_iterable(include):
                include = (include,)
            order = filter(include.__contains__, order)

    if exclude:
        if isinstance(exclude, FunctionType | LambdaType):
            order = filterfalse(exclude, order)
        else:
            if not is_iterable(exclude):
                exclude = (exclude,)
            order = filterfalse(exclude.__contains__, order)

    if exclude_stdlib:
        order = filterfalse(is_internal, order)

    yield from order


def _get_attribute_from_inheritance(something, name, **kw):

    index = kw.pop('index', 0)
    kw.setdefault('exclude_self', False)
    kw.setdefault('exclude_stdlib', False)

    counter = 0
    for obj in iter_inheritance(something, **kw):
        try:
            attr = obj.__dict__[name]

        except KeyError:
            continue

        if not counter - index:
            return attr, obj
        counter += 1

    raise KeyError(name)


def get_owner(obj, name, **kw):
    with suppress(KeyError):
        return _get_attribute_from_inheritance(obj, name, **kw)[1]


def get_attr(obj, name, default=None, **kw):
    try:
        return _get_attribute_from_inheritance(obj, name, **kw)[0]
    except KeyError:
        return default


def to_bytes(x, charset=None) -> bytes:
    if not isinstance(x, bytes | str):
        msg = f'only bytes | str acceptable, not {just_value(x)}'
        raise TypeError(msg)
    return x.encode(charset or 'ascii') if isinstance(x, str) else x


def to_ascii(x, *args, **kw) -> str:
    charset = kw.pop('charset', 'ascii')
    if not isinstance(x, bytes | str):
        msg = f'only bytes | str acceptable, not {just_value(x)}'
        raise TypeError(msg)
    return x if isinstance(x, str) else to_bytes(x, *args, **kw).decode(charset)


def unique(iterable, /, key=None, include=None, exclude=None):
    skip = include is None

    if not key:
        exclude = set(exclude or ())
        include = frozenset(include or ())

    else:
        exclude = set(map(key, exclude or ()))
        include = frozenset(map(key, include or ()))

    excluded = exclude.__contains__
    included = include.__contains__
    is_dict = is_mapping(iterable)

    for element in iterable:

        k = key(element) if key else element
        if not excluded(k) and (skip or included(k)):

            yield (element, iterable[element]) if is_dict else element
            exclude.add(k)
