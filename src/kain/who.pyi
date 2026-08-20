from collections.abc import Callable
from functools import partial
from typing import Any

__all__ = (
    "Addr",
    "Args",
    "Cast",
    "File",
    "Inheritance",
    "Is",
    "Module",
    "Name",
)

Args: Callable[..., str]
Cast: Callable[..., str]
File: Callable[..., str | None]
Inheritance: Callable[..., tuple[Any, ...] | str]
Is: Callable[..., str]
Module: Callable[..., str]
Addr: partial[str]
Name: partial[str]
