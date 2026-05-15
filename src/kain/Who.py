from collections.abc import (
    Callable,
)
from functools import partial
from typing import Any

from kain.internals import (
    format_args_and_keywords,
    get_mro,
    just_value,
    pretty_module,
    source_file,
    who_is,
)

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

Args: Callable[..., str] = format_args_and_keywords
Cast: Callable[..., str] = just_value
File: Callable[..., str | None] = source_file
Inheritance: Callable[..., tuple[Any, ...] | str] = get_mro
Is: Callable[..., str] = who_is
Module: Callable[..., str] = pretty_module
Addr: partial[str] = partial(who_is, addr=True)
Name: partial[str] = partial(who_is, full=False)
