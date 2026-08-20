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
from contextlib import suppress
from functools import cache
from importlib import import_module
from inspect import ismodule
from logging import getLogger
from os import sep
from pathlib import Path
from types import ModuleType

from kain import _who
from kain.internals import iter_stack, to_ascii, unique

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

SINGLE_COMPONENT = 2


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

    if count == SINGLE_COMPONENT:
        with suppress(ModuleNotFoundError):
            return import_module(path), ()

    chunk = path
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
                f"{path} (object {_who.Is(parent)} hasn't attribute "
                f"{child!a}{_who.File(parent, ' in %a') or ''})",
            )

        if not set(dir(parent)) - IGNORED_OBJECT_FIELDS:
            chunk = f"{_who.Is(parent)}.{child}"
            raise ImportError(
                f"{path} (from partially initialized module "
                f"{chunk!a}, most likely due to a circular import"
                f'{_who.File(parent, " from %a") or ""}) or just not found',
            )

        raise ImportError(
            f"{path} (module {_who.Is(parent)} hasn't member {child!a}"
            f"{_who.File(parent, ' in %a') or ''})",
        )

    return getattr(parent, child)


def import_object(path: str | bytes) -> object:
    """Dynamically import an object by its fully-qualified name.

    Supports importing:
    - Entire modules (``os``, ``kain.importer``)
    - Module attributes (``os.path.join``, ``kain.importer.required``)

    Args:
        path: A dotted import path as string or bytes.

    Returns:
        The imported object (module, class, function, etc.).

    Raises:
        TypeError: If ``path`` is not a string or bytes.
        ImportError: If the module or attribute cannot be found.

    Example:
        >>> import_object("os.path.join")
        <function join at ...>
    """
    if not isinstance(path, str | bytes):
        raise TypeError(f"{_who.Is(path)} isn't str")

    path = to_ascii(path)
    logger.debug(f"lookup: {path}")

    something, sequence = get_module(path)
    if something is None:  # type: ignore[redundant-expr]  # pyright: ignore[reportUnnecessaryComparison]
        raise ImportError(f"{path} (isn't exists?)")

    locator = str(path)
    if not sequence:
        logger.debug(f"import path: {_who.Is(something)}")

    else:
        logger.debug(
            f"split path: {_who.Is(something)} (module) "
            f'-> {".".join(sequence)} (path)',
        )

    for name in sequence:
        something = get_child(
            locator,
            something,
            name,
        )

    logger.debug("load ok: %s", path)
    return something


@cache
def cached_import(path: str | bytes) -> object:
    """Cached version of ``import_object``.

    Uses :func:`functools.cache` to memoize import results. Subsequent
    calls with the same path return the cached result.

    Args:
        path: The import path passed to ``import_object``.

    Returns:
        The imported (and cached) object.

    Example:
        >>> cached_import("os.path.join")
    """
    return import_object(path)


def required(
    path: str,
    *,
    throw: bool = True,
    quiet: bool = False,
    default: object = None,
) -> object:
    """Import an object, requiring it to exist.

    Attempts to import the object at ``path``. If the import fails,
    behavior is controlled by the ``throw``, ``quiet``, and ``default``
    parameters.

    Args:
        path: The import path (e.g., ``os.path.join``).
        throw: If True (default), raise ImportError on failure.
            If False, return ``default`` on failure.
        quiet: If True, suppress warning log on failure.
            If False (default), log a warning on failure.
        default: Value to return on failure when ``throw=False``.

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
    try:
        try:
            return cached_import(path)

        except TypeError:
            return import_object(path)

    except ImportError as e:

        if not quiet or throw:
            msg = f"couldn't import required({path=})"

            base = path.split(".", 1)[0]
            if base not in sys.modules:
                package = (PACKAGES_MAP.get(base) or base).replace("_", "-")
                msg = f"{msg}; (need extra package={package!r})"

            if not quiet:
                logger.warning(msg)

            if throw:
                raise ImportError(msg) from e

    return default


def optional(
    path: str,
    *,
    default: object = None,
    throw: bool = False,
    quiet: bool = True,
) -> object:
    """Import an object optionally, returning None on failure.

    Convenience wrapper around ``required`` with ``quiet=True``
    and ``throw=False`` by default.

    Args:
        path: The import path.
        default: Value to return on failure.
        throw: If True, raise ImportError on failure.
            Defaults to False.
        quiet: If True, suppress warning log on failure.
            Defaults to True.

    Returns:
        The imported object, or ``default`` if specified and import failed,
        or None if import failed and no default specified.

    Raises:
        ImportError: If import fails and ``throw=True``.

    Example:
        >>> optional("natsort.natsorted", default=sorted)
        <built-in function sorted>
        >>> optional("nonexistent_module")
        None
    """
    return required(path, throw=throw, quiet=quiet, default=default)


def get_path(  # noqa: PLR0912
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
        file = ""
        for file in iter_stack(1, offset=1):
            if Path(file).stem != base:
                break
        root = Path(file).parent

    if isinstance(root, Path | str):  # type: ignore[redundant-expr]  # pyright: ignore[reportUnnecessaryIsInstance]
        root = Path(root)
    else:
        raise TypeError(
            f"root={root!r} can be str | {_who.Is(Path)} | None, "
            f"not {_who.Is(root)}",
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
        root = get_path(path, **kw)  # type: ignore[arg-type]  # pyrefly: ignore[bad-argument-type]  # pyright: ignore[reportArgumentType]
        if not root:  # type: ignore[truthy-bool]
            raise ValueError(f"{path=} not found, {_who.Args(**kw)}")
        path = root if str(path).startswith(".") else (root / path).resolve()

    str_path = str(path.resolve())
    if str_path not in sys.path:
        sys.path.append(str_path)
        sys.path = list(unique(sys.path))
        exists = path.is_dir()
        logger.info(f"path {request} resolved to {path}, {exists=}")
    return path
