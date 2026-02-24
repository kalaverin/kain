"""Dynamic import utilities and path management for kain.

This module provides utilities for:
- Dynamic importing of modules, classes, functions, and attributes
- Safe optional imports with fallback defaults
- Path resolution and ``sys.path`` manipulation

The main functions are:
    - required(): Import with mandatory success or exception
    - optional(): Import with graceful fallback to default
    - add_path(): Add resolved paths to ``sys.path``

Example:
    >>> from kain.importer import required, optional, add_path
    >>> os_path = required('os.path')
    >>> natsort = optional('natsort.natsorted', default=sorted)
    >>> add_path('..')  # Add parent directory to sys.path
"""

import sys
from collections.abc import Callable
from contextlib import suppress
from functools import cache
from importlib import import_module
from inspect import ismodule
from logging import getLogger
from os import sep
from pathlib import Path
from types import ModuleType
from typing import overload

from kain.internals import Who, iter_stack, to_ascii, unique

__all__ = ("add_path", "optional", "required")

logger = getLogger(__name__)

#: Module attribute names to ignore when checking for circular imports.
#: These are standard dunder attributes present on all modules.
IGNORED_OBJECT_FIELDS: set[str] = {
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

#: Mapping of import names to their PyPI package names.
#: Used to provide helpful error messages when optional dependencies
#: are not installed.
PACKAGES_MAP: dict[str, str] = {"magic": "python-magic", "git": "gitpython"}


@cache
def get_module(path: str) -> tuple[ModuleType, tuple[str, ...]]:
    """Import a module and return remaining attribute path components.

    Attempts to import the longest possible module prefix from a dot-
    separated path, returning the imported module and any remaining
    attribute path components.

    Args:
        path: Dot-separated import path (e.g., ``os.path.join``).

    Returns:
        A tuple of ``(module, remaining_path_components)`` where
        ``module`` is the imported module and ``remaining_path_components``
        is a tuple of attribute names to traverse.

    Raises:
        ImportError: If no module prefix can be imported.

    Example:
        >>> get_module("os.path.join")
        (<module 'posixpath' ...>, ('join',))
        >>> get_module("kain.importer")
        (<module 'kain.importer' ...>, ())
    """
    chunks = path.split(".")
    count = len(chunks) + 1

    if count == 2:  # noqa: PLR2004
        with suppress(ModuleNotFoundError):
            return import_module(path), ()

    for i in range(1, count):
        chunk = ".".join(chunks[: count - i])
        with suppress(ModuleNotFoundError):
            return import_module(chunk), tuple(chunks[count - i :])

    msg = f"ImportError: {path} ({chunk!a} does not exist)"
    raise ImportError(msg)


def get_child(path: str, parent: object, child: str) -> object:
    """Get an attribute from a parent object with enhanced error messages.

    If ``parent`` is a module, attempts to import ``child`` from it first
    using ``__import__`` to ensure submodules are loaded.

    Args:
        path: The full import path (for error messages).
        parent: The object to get the attribute from.
        child: The attribute name to retrieve.

    Returns:
        The attribute value.

    Raises:
        ImportError: If the attribute doesn't exist. The error message
        includes context about whether the parent is a module, and may
        suggest a circular import if the module appears to be partially
        initialized (has no public attributes).
    """
    if ismodule(parent):
        __import__(parent.__name__, globals(), locals(), [str(child)])

    if not hasattr(parent, child):
        if not ismodule(parent):
            raise ImportError(
                f"{path} (object {Who.Is(parent)} hasn't attribute "
                f"{child!a}{Who.File(parent, ' in %a') or ''})",
            )

        if not set(dir(parent)) - IGNORED_OBJECT_FIELDS:
            chunk = f"{Who.Is(parent)}.{child}"
            raise ImportError(
                f"{path} (from partially initialized module "
                f"{chunk!a}, most likely due to a circular import"
                f'{Who.File(parent, " from %a") or ""}) or just not found',
            )

        raise ImportError(
            f"{path} (module {Who.Is(parent)} hasn't member {child!a}"
            f"{Who.File(parent, ' in %a') or ''})",
        )

    return getattr(parent, child)


@overload
def import_object(path: str | bytes, something: None = None) -> object: ...


@overload
def import_object(path: str | bytes, something: object) -> object: ...


def import_object(
    path: str | bytes | None,
    something: object = None,
) -> object:
    """Dynamically import an object by its fully-qualified name.

    Supports importing:
    - Entire modules (``os``, ``kain.importer``)
    - Module attributes (``os.path.join``, ``kain.importer.required``)
    - Attributes from given parent objects

    Args:
        path: Can be a string or bytes. If ``something`` is provided and
            ``path`` is not a string, the arguments are swapped so that
            ``something`` is treated as the import path.
        something: Optional parent object or import path string. If
            ``path`` is a string, this is treated as the parent object. If
            ``path`` is not a string, this is treated as the import path
            string and the arguments are swapped.

    Returns:
        The imported object (module, class, function, etc.).

    Raises:
        TypeError: If both arguments are None, or if path is not a string
            and something is None.
        ImportError: If the module or attribute cannot be found.

    Example:
        >>> import_object("os.path.join")
        <function join at ...>
        >>> import_object("path.join", os)
        <function join at ...>
    """
    if path is something is None:
        raise TypeError("all arguments are None")

    if isinstance(path, str | bytes):
        path = to_ascii(path)

    if not isinstance(path, str):
        if something is None:
            msg = (
                f"{Who.Is(path)} isn't str, but "
                f"second argument (import path) is None"
            )
            raise TypeError(msg)
        path, something = something, path

    logger.debug(f"lookup: {path}")

    if something:
        locator = f"{Who.Is(something)}.{path}"
        sequence = path.split(".")

    else:
        locator = str(path)
        something, sequence = get_module(path)

        if something is None:
            raise ImportError(f"{path} (isn't exists?)")

    if not sequence:
        logger.debug(f"import path: {Who.Is(something)}")

    else:
        logger.debug(
            f'split path: {Who.Is(something)} (module) -> {".".join(sequence)} (path)',
        )

    for name in sequence:
        something = get_child(locator, something, name)

    logger.debug("load ok: %s", path)
    return something


@cache
def cached_import(*args: object, **kw: object) -> object:
    """Cached version of ``import_object``.

    Uses :func:`functools.cache` to memoize import results. Subsequent
    calls with the same arguments return the cached result.

    Args:
        *args: Positional arguments passed to ``import_object``.
        **kw: Keyword arguments passed to ``import_object``.

    Returns:
        The imported (and cached) object.

    Example:
        >>> cached_import("os.path.join")
    """
    return import_object(*args, **kw)


def required(path: str, *args: object, **kw: object) -> object:
    """Import an object, requiring it to exist.

    Attempts to import the object at ``path``. If the import fails,
    behavior is controlled by the ``throw``, ``quiet``, and ``default``
    parameters.

    Args:
        path: The import path (e.g., ``os.path.join``).
        *args: Additional positional arguments passed to ``cached_import`` /
            ``import_object``.
        throw: If True (default), raise ImportError on failure.
            If False, return ``default`` on failure.
        quiet: If True, suppress warning log on failure.
            If False (default), log a warning on failure.
        default: Value to return on failure when ``throw=False``.
        **kw: Additional keyword arguments passed to ``cached_import`` /
            ``import_object``.

    Returns:
        The imported object, or ``default`` if import failed and
        ``throw=False``.

    Raises:
        ImportError: If import fails and ``throw=True``.

    Example:
        >>> required("os.path.join")
        <function join at ...>
        >>> required("nonexistent", throw=False, default="fallback")
        'fallback'
    """
    throw = kw.pop("throw", True)
    quiet = kw.pop("quiet", False)
    default: object = kw.pop("default", None)

    try:
        try:
            return cached_import(path, *args, **kw)

        except TypeError:
            return import_object(path, *args, **kw)

    except ImportError as e:

        if not quiet or throw:
            msg = f"couldn't import required({path=}, *{args=}, **{kw=})"

            base = path.split(".", 1)[0]
            if base not in sys.modules:
                package = (PACKAGES_MAP.get(base) or base).replace("_", "-")
                msg = f"{msg}; (need extra package={package!r})"

            if not quiet:
                logger.warning(msg)

            if throw:
                raise ImportError(msg) from e

    return default


def optional(path: str, *args: object, **kw: object) -> object:
    """Import an object optionally, returning None on failure.

    Convenience wrapper around ``required`` with ``quiet=True``
    and ``throw=False`` by default.

    Args:
        path: The import path.
        *args: Additional positional arguments passed to ``required``.
        default: Value to return on failure.
        **kw: Additional keyword arguments passed to ``required``.
            Defaults: ``quiet=True``, ``throw=False``.

    Returns:
        The imported object, or ``default`` if specified and import failed,
        or None if import failed and no default specified.

    Raises:
        ImportError: If import fails and ``throw=True`` is passed in **kw.

    Example:
        >>> optional("natsort.natsorted", default=sorted)
        <built-in function sorted>
        >>> optional("nonexistent_module")
        None
    """
    kw.setdefault("quiet", True)
    kw.setdefault("throw", False)
    return required(path, *args, **kw)


#: Natural sort function if ``natsort`` is installed, otherwise
#: falls back to built-in :func:`sorted`.
sort: Callable[..., list[object]] = optional(
    "natsort.natsorted",
    quiet=True,
    default=sorted,
)


def get_path(
    path: str | Path,
    root: str | Path | None = None,
) -> Path:
    """Resolve a path relative to a root directory.

    Supports multiple path formats:
    - ``.``: Returns as-is (current directory reference)
    - ``..``, ``...``, etc.: Go up N-1 parent directories from root
    - ``../foo``: Resolve relative to root
    - ``subdir/name``: If ``path`` is a substring of ``root``, return the
        prefix of ``root`` up to the first occurrence of ``path``.
    - ``dirname``: Walk up from root looking for directory name

    Args:
        path: The path to resolve. Can be string or Path.
        root: The root directory to resolve from. If None, uses the
            directory of the calling module.

    Returns:
        The resolved absolute Path.

    Example:
        >>> get_path("..", "/project/src")
        '/project'
        >>> get_path("src", "/project/src/module.py")
        '/project/src'

    Raises:
        TypeError: If root is not str, Path, or None.
        ValueError: If the path cannot be resolved.
    """
    if root is None:

        base = Path(__file__).stem
        for file in iter_stack(1, offset=1):
            if Path(file).stem != base:
                break
        root = Path(file).parent

    if isinstance(root, Path | str):
        root = Path(root)
    else:
        raise TypeError(
            f"root={root!r} can be str | {Who.Is(Path)} | None, not {Who.Is(root)}",
        )

    spath = str(path).strip("/")

    if set(spath) == {"."}:
        dots = len(spath) - 2
        if dots == -1:
            return Path(path)

        path = root.resolve()
        for _ in range(dots + 1):
            path = path.parent

        return path.resolve()

    if spath.startswith("../"):
        return (root / path).resolve()

    if sep in str(path) and ("../" not in spath and "/.." not in spath):
        try:
            idx = str(root).index(str(path))
        except ValueError as e:
            raise ValueError(f"{path=} not found in {root=}") from e
        return Path(str(root)[:idx])

    subdir = str(root)
    while subdir != sep:

        future = Path(subdir)
        subdir = str(future.parent)

        if path == future.name:
            return future

    raise ValueError(f"{path=} not found in {root=}")


def add_path(path: str | Path, **kw: object) -> Path:
    """Add a resolved path to ``sys.path``.

    Resolves the given path and adds it to ``sys.path`` if not already
    present. Handles relative paths, file paths, and dot notation for
    parent directories.

    Args:
        path: The path to add. Can be:
            - ``..``, ``...``, etc. - parent directories
            - A file path - adds the parent directory
            - A relative path - resolved using :func:`get_path`
            - An absolute path - used as-is
        **kw: Additional arguments passed to :func:`get_path`.

    Returns:
        The resolved Path that was added (or was already present).

    Raises:
        ValueError: If the path cannot be resolved.

    Example:
        >>> add_path('..')      # Add parent directory
        PosixPath('/home/user/project')
        >>> add_path('src')     # Resolve and add 'src' directory
        PosixPath('/home/user/project/src')
    """
    path = Path(path)
    request = path

    if path.is_file():
        path = path.resolve().parent

    elif not (str(path).startswith(sep) or path == path.resolve()):
        root = get_path(path, **kw)
        if not root:
            raise ValueError(f"{path=} not found, {Who.Args(**kw)}")
        path = root if str(path).startswith(".") else (root / path).resolve()

    str_path = str(path.resolve())
    if str_path not in sys.path:
        sys.path.append(str_path)
        sys.path = list(unique(sys.path))
        exists = path.is_dir()
        logger.info(f"path {request} resolved to {path}, {exists=}")
    return path
