# ruff: noqa: ANN401, FBT001, FBT002

import sys
from collections import deque
from collections.abc import (
    Callable,
    Collection,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
)
from contextlib import suppress
from dataclasses import dataclass
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
from types import FunctionType, GenericAlias, LambdaType, ModuleType, UnionType
from typing import Any, TypeVar, cast, get_args, get_origin, overload

Collections: tuple[type, ...] = deque, dict, list, set, tuple, bytearray
"""Built-in mutable and immutable collection types."""

Primitives: tuple[type, ...] = bool, float, int, str, complex, bytes
"""Primitive scalar types."""

Builtins: tuple[type, ...] = Primitives + Collections
"""All built-in primitive and collection types."""

WinNT: bool = "windows" in architecture()[1].lower()
"""True when running on a Windows NT platform."""

__all__ = (
    "Is",
    "Who",
    "get_attr",
    "get_owner",
    "iter_inheritance",
    "iter_stack",
    "to_ascii",
    "to_bytes",
    "unique",
)


def class_of(obj: Any) -> type[Any]:
    """Return the class of *obj*.

    If *obj* is already a class, it is returned unchanged.
    """
    return obj if isclass(obj) else type(obj)


def is_callable(obj: Any) -> bool:
    """Return whether *obj* is callable."""
    return isinstance(obj, Callable) or callable(obj)  # type: ignore[arg-type]


def is_collection(obj: Any) -> bool:
    """Return whether *obj* looks like a collection.

    Sequences (except ``str``/``bytes``), mappings, and duck-typed
    objects implementing ``__getitem__``, ``__setitem__`` and
    ``__delitem__`` are considered collections.
    """
    return (
        (
            isinstance(obj, Collection)
            and isinstance(obj, Sequence)
            and not isinstance(obj, bytes | str)
        )
        or is_mapping(obj)
        or all(
            map(
                partial(hasattr, obj),
                ("__getitem__", "__setitem__", "__delitem__"),
            ),
        )
    )


def is_iterable(obj: Any) -> bool:
    """Return whether *obj* is iterable."""
    return isinstance(obj, Iterable) or hasattr(obj, "__iter__")


def is_mapping(obj: Any) -> bool:
    """Return whether *obj* is a mapping."""
    return isinstance(obj, Mapping) or issubclass(class_of(obj), dict)


def is_primitive(obj: Any) -> bool:
    """Return whether *obj* is a primitive or built-in container."""
    return obj is True or obj is False or obj is None or type(obj) in Builtins


def is_from_primivite(obj: Any) -> bool:
    """Return whether *obj* is ``None`` or a primitive instance."""
    return bool(obj is None or isinstance(obj, Primitives))


def is_from_builtin(obj: Any) -> bool:
    """Return whether *obj* is ``None`` or a built-in instance."""
    return bool(isinstance(obj, Collections) or is_from_primivite(obj))


def is_interactive() -> bool:
    """Return whether the current session is attached to a TTY."""
    if not getattr(sys, "frozen", False):  # nuitka compiler checks this
        return all(map(methodcaller("isatty"), (stderr, stdin, stdout)))
    return False


@cache
def _get_module_path_type(full: Any) -> tuple[bool | None, str]:
    """Categorise *full* as stdlib, local project, or external.

    Returns a ``(is_stdlib, relative_or_full_path)`` tuple where
    ``is_stdlib`` is ``True`` for the standard library, ``False`` for
    the current project or site-packages, and ``None`` for unknown
    paths.
    """
    dirs = get_paths()

    path = str(full)
    if WinNT:
        path = path.lower()

    for scheme, reason in (
        ("stdlib", True),
        ("purelib", False),
        ("platlib", False),
        ("platstdlib", True),
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


def is_internal(x: Any) -> bool:
    """Return whether *x* originates from the standard library."""
    if isbuiltin(x) or isbuiltin(class_of(x)):
        return True

    if module := get_module(x):

        if module.__name__ == "builtins":
            return True

        file_path = getattr(module, "__file__", None)
        if file_path is None:
            return True

        is_stdlib = _get_module_path_type(file_path)[0]
        if is_stdlib is not None:
            return is_stdlib

    return False


def is_subclass(obj: Any, types: Any) -> bool:  # noqa: PLR0911
    """Extended subclass check supporting typing constructs.

    Handles ``Any``, ``UnionType``, ``GenericAlias``, and plain
    classes.  Returns ``False`` when *types* is ``None``.
    """
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

        if origin is UnionType and (
            issubclass(cls, types) or (args and cls in args)
        ):
            return True

        if class_of(types) is GenericAlias:
            # dict[str, str])
            return issubclass(cls, origin)

    return issubclass(cls, types)


def get_module(x: Any) -> ModuleType | None:
    """Resolve the module that defines *x*."""
    if ismodule(x):
        return x

    if (module := getmodule(x)) or (module := getmodule(class_of(x))):
        return module

    return None


def get_module_name(x: Any) -> str | None:
    """Return the dotted module name for *x*."""
    if module := get_module(x):
        with suppress(AttributeError):
            if spec := module.__spec__:
                return spec.name
    return None


def object_name(obj: Any, full: bool = True) -> str:
    """Build a readable, fully-qualified name for *obj*."""
    def post(x: str) -> str:
        return sub(
            r"^([\?\.]+)",
            "",
            sub("^(__main__|__builtin__|builtins)", "", x),
        )

    def get_module_from(x: Any) -> str:
        return getattr(x, "__module__", get_module_name(x)) or "?"

    def get_object_name(x: Any) -> str:
        if x is Any:
            return "typing.Any" if full else "Any"

        name: str = getattr(
            x,
            "__qualname__",
            getattr(x, "__name__", str(x)),
        )
        module = get_module_from(x)

        if not name.startswith(module):
            name = f"{module}.{name}"
        return name

    def main(obj: Any) -> str:
        if ismodule(obj):
            return get_module_name(obj) or "?"

        for itis in iscoroutine, isfunction, ismethod:
            if itis(obj):
                name = get_object_name(obj)
                with suppress(AttributeError):
                    self_or_cls = (
                        obj.im_self  # type: ignore[attr-defined]
                        or obj.im_class  # type: ignore[attr-defined]
                    )
                    name = f"{object_name(self_or_cls)}.{post(name)}"
                return name

        cls = class_of(obj)
        if cls in (property, classmethod, staticmethod):
            if cls is property:
                return get_object_name(obj.fget)
            return get_object_name(cls)

        if (hasattr(obj, "__qualname__") or hasattr(obj, "__name__")) and (
            not isclass(obj) and not ismodule(obj)
        ):
            return get_object_name(obj)

        return get_object_name(cls)

    name = post(main(obj))
    return name if full else name.rsplit(".", 1)[-1]


def pretty_module(obj: Any) -> str:
    """Return the module portion of *obj*'s fully-qualified name."""
    return who_is(obj).rsplit(".", 1)[0]


def source_file(
    obj: Any,
    template: str | None = None,
    **kw: Any,
) -> str | None:
    """Locate the source file of *obj*'s class through inheritance."""
    kw.setdefault("exclude_self", False)
    kw.setdefault("exclude_stdlib", False)

    for child in iter_inheritance(class_of(obj), **kw):
        with suppress(TypeError):
            if path := getsourcefile(child):
                return (template % path) if template else str(path)
    return None


def just_value(obj: Any, /, **kw: Any) -> str:
    """Return a short ``(Type)value`` representation of *obj*."""
    kw.setdefault("addr", False)
    name = who_is(obj, **kw)

    return f"({name})" if isclass(obj) else f"({name}){obj}"


@cache
def is_imported_module(name: str) -> bool:
    """Return whether *name* (or its parent package) is in
    ``sys.modules``.
    """
    with suppress(KeyError):
        return bool(modules[name])

    chunks = name.split(".")
    return (
        sum(".".join(chunks[: no + 1]) in modules for no in range(len(chunks)))
        >= 2  # noqa: PLR2004
    )


def get_mro(obj: Any, /, **kw: Any) -> tuple[Any, ...] | str:
    """Walk the inheritance chain of *obj* with optional mapping.

    Accepts ``func`` to transform each class and ``glue`` to join
    the result into a string.
    """
    func: Callable[[Any], Any] | None = kw.pop("func", None)
    glue: str | None = kw.pop("glue", None)

    result = iter_inheritance(obj, **kw)

    if func:
        mapped = tuple(map(func, result))
        if glue:
            return glue.join(mapped)
        return mapped

    if glue:
        return glue.join(result)

    return tuple(result)


def simple_repr(x: Any) -> bool | str | None:
    """Return a plain Python literal when possible.

    Falls back to a type description via :func:`just_value` for
    complex objects.
    """
    if (x is None or x is True or x is False) or isinstance(x, str):
        return x

    if isinstance(x, int | float):
        return repr(x)

    return just_value(x)


def format_args_and_keywords(*args: Any, **kw: Any) -> str:
    """Pretty-print positional and keyword arguments."""
    def format_args(x: tuple[Any, ...]) -> str:
        return repr(tuple(map(simple_repr, x)))[1:-1].rstrip(",")

    def format_kwargs(x: dict[str, Any]) -> str:
        return ", ".join(f"{k}={simple_repr(v)}" for k, v in x.items())

    if args and kw:
        return f"{format_args(args)}, {format_kwargs(kw)}"

    if args:
        return format_args(args)

    if kw:
        return format_kwargs(kw)

    return ""


# public interface, Is/Who


def who_is(obj: Any, /, full: bool = True, addr: bool = False) -> str:
    """Return the fully-qualified name of *obj*.

    Caches the result on the object itself.  When *addr* is ``True``,
    appends the ``id()`` in hex.
    """
    key = "__name_full__" if full else "__name_short__"

    def get_name() -> str:
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
    if not addr:
        return name
    return f"{name}#{id(obj):x}"


@dataclass
class Who:
    """Namespace holding introspection helpers."""

    Args: Callable[..., str] = format_args_and_keywords
    Cast: Callable[..., str] = just_value
    File: Callable[..., str | None] = source_file
    Inheritance: Callable[..., tuple[Any, ...] | str] = get_mro
    Is: Callable[..., str] = who_is
    Module: Callable[..., str] = pretty_module
    Addr: partial[str] = partial(who_is, addr=True)  # noqa: RUF009
    Name: partial[str] = partial(who_is, full=False)  # noqa: RUF009


@dataclass
class Is:
    """Namespace holding type-check predicates."""

    Builtin: Callable[..., bool] = is_from_builtin
    Class: Callable[..., Any] = isclass

    Primivite: Callable[..., bool] = is_from_primivite
    tty: bool = is_interactive()
    awaitable: Callable[..., bool] = isawaitable
    builtin: Callable[..., bool] = isbuiltin
    callable: Callable[..., bool] = is_callable
    classOf: Callable[..., type[Any]] = class_of  # noqa: N815
    collection: Callable[..., bool] = is_collection
    coroutine: Callable[..., bool] = iscoroutine
    function: Callable[..., bool] = isfunction
    imported: Callable[..., bool] = is_imported_module
    internal: Callable[..., bool] = is_internal
    iterable: Callable[..., bool] = is_iterable
    mapping: Callable[..., bool] = is_mapping
    method: Callable[..., bool] = ismethod
    module: Callable[..., bool] = ismodule
    primitive: Callable[..., bool] = is_primitive
    subclass: Callable[..., bool] = is_subclass


# public functions


def iter_stack(*args: Any, **kw: Any) -> Iterator[Any]:
    """Yield frames from the current call stack.

    *args* are attribute names extracted from each frame.
    *offset* skips the first *n* frames.
    """
    result = stack()[kw.pop("offset", 0) :]
    yield from (map(itemgetter(*args), result) if args else result)


def iter_inheritance(  # noqa: PLR0913
    obj: Any,
    include: Any = None,
    exclude: Any = None,
    exclude_self: bool = True,
    exclude_stdlib: bool = True,
    reverse: bool = False,
) -> Iterator[Any]:
    """Yield the MRO of *obj*'s class with configurable filtering."""
    order: Iterator[Any]
    mro = class_of(obj).__mro__[:-1]

    if not exclude_self:
        order = unique((obj, *mro), key=id)
    else:
        order = unique(filter(lambda x: x is not obj, mro), key=id)

    if reverse:
        order = iter(reversed(list(order)))

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


def _get_attribute_from_inheritance(
    obj: Any,
    name: str,
    **kw: Any,
) -> tuple[Any, Any]:
    """Locate *name* along the inheritance chain.

    Returns ``(attribute, owner)``.  *index* selects the n-th match.
    """
    index: int = kw.pop("index", 0)
    kw.setdefault("exclude_self", False)
    kw.setdefault("exclude_stdlib", False)

    counter = 0
    for child in iter_inheritance(obj, **kw):
        try:
            attr = child.__dict__[name]

        except KeyError:
            continue

        if not counter - index:
            return attr, child
        counter += 1

    raise KeyError(name)


def get_owner(obj: Any, name: str, **kw: Any) -> Any | None:
    """Return the class that owns *name* in *obj*'s MRO."""
    with suppress(KeyError):
        return _get_attribute_from_inheritance(obj, name, **kw)[1]
    return None


def get_attr(obj: Any, name: str, default: Any = None, **kw: Any) -> Any:
    """Fetch *name* from *obj*'s inheritance chain.

    Falls back to *default* when the attribute is missing.
    """
    try:
        return _get_attribute_from_inheritance(obj, name, **kw)[0]
    except KeyError:
        return default


def to_ascii(x: bytes | str, /, charset: str | None = None) -> str:
    """Coerce *x* to a ``str`` using the given *charset*."""
    if not isinstance(x, bytes | str):
        raise TypeError(f"only bytes | str acceptable, not {just_value(x)}")

    if isinstance(x, str):
        return x

    charset = charset or "ascii"
    return to_bytes(x, charset=charset).decode(charset)


def to_bytes(x: bytes | str, /, charset: str | None = None) -> bytes:
    """Coerce *x* to ``bytes`` using the given *charset*."""
    if not isinstance(x, bytes | str):
        raise TypeError(f"only bytes | str acceptable, not {just_value(x)}")

    if not isinstance(x, str):
        return x

    return x.encode(charset or "ascii")


T = TypeVar("T")
K = TypeVar("K")
V = TypeVar("V")
K_cmp = TypeVar("K_cmp")


# Mapping без key — сравниваем по ключам, возвращаем пары (key, value)
@overload
def unique(
    iterable: Mapping[K, V],
    /,
    key: None = None,
    include: Iterable[K] | None = None,
    exclude: Iterable[K] | None = None,
) -> Iterator[tuple[K, V]]: ...


# Mapping с key — ключи трансформируются для сравнения
@overload
def unique(
    iterable: Mapping[K, V],
    /,
    key: Callable[[K], K_cmp],
    include: Iterable[K_cmp] | None = None,
    exclude: Iterable[K_cmp] | None = None,
) -> Iterator[tuple[K, V]]: ...


# Обычный Iterable без key
@overload
def unique(
    iterable: Iterable[T],
    /,
    key: None = None,
    include: Iterable[T] | None = None,
    exclude: Iterable[T] | None = None,
) -> Iterator[T]: ...


# Обычный Iterable с key
@overload
def unique(
    iterable: Iterable[T],
    /,
    key: Callable[[T], K_cmp],
    include: Iterable[K_cmp] | None = None,
    exclude: Iterable[K_cmp] | None = None,
) -> Iterator[T]: ...


def unique(
    iterable: Iterable[Any],
    /,
    key: Callable[[Any], Any] | None = None,
    include: Iterable[Any] | None = None,
    exclude: Iterable[Any] | None = None,
) -> Iterator[Any]:
    """Yield unique elements from *iterable* based on an optional *key*.

    For mappings yields ``(key, value)`` pairs.  Already-seen keys
    (or elements when *key* is omitted) are skipped.
    """
    skip = include is None

    if not key:
        exclude_set: set[Any] = set(exclude or ())
        include_set: frozenset[Any] = frozenset(include or ())
    else:
        exclude_set = set(map(key, exclude or ()))
        include_set = frozenset(map(key, include or ()))

    excluded = exclude_set.__contains__
    included = include_set.__contains__
    is_dict = is_mapping(iterable)

    for element in iterable:  # type: ignore[attr-defined]
        k = key(element) if key else element  # type: ignore[arg-type]
        if not excluded(k) and (skip or included(k)):
            yield (
                (element, cast("Mapping[Any, Any]", iterable)[element])
                if is_dict
                else element
            )
            exclude_set.add(k)
