from pathlib import Path
from types import ModuleType

__all__ = ("add_path", "optional", "required")

def get_module(path: str) -> tuple[ModuleType, tuple[str, ...]]: ...
def get_child(path: str, parent: object, child: str) -> object: ...
def import_object(path: str | bytes) -> object: ...
def cached_import(path: str | bytes) -> object: ...
def required(
    path: str,
    *,
    throw: bool = True,
    quiet: bool = False,
    default: object = None,
) -> object: ...
def optional(
    path: str,
    *,
    default: object = None,
    throw: bool = False,
    quiet: bool = True,
) -> object: ...
def get_path(
    path: str | Path,
    root: str | Path | None = None,
) -> Path: ...
def add_path(path: str | Path, **kw: object) -> Path: ...
