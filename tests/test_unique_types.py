"""Static type assertions for kain.internals.unique.

Run ``basedpyright tests/test_unique_types.py`` or
``basedmypy tests/test_unique_types.py`` to verify.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, assert_type

from kain.internals import unique


def test_type_inference_dummy() -> None:
    """Runtime placeholder; real checking is done by static analyzers."""


if TYPE_CHECKING:
    # Without key - types inferred from the iterable.
    result1 = unique([1, 2, 3])
    assert_type(
        result1,
        Iterator[int],
    )  # pyright: ignore[reportUnusedCallResult]

    result2 = unique("abc")
    assert_type(
        result2,
        Iterator[str],
    )  # pyright: ignore[reportUnusedCallResult]

    # With key - element type stays the same, key return type is separate.
    result3 = unique(["a", "bb"], key=len)
    assert_type(
        result3,
        Iterator[str],
    )  # pyright: ignore[reportUnusedCallResult]

    # With include/exclude and no key - filter values match element type.
    result4 = unique([1, 2, 3], include=[1, 2])
    assert_type(
        result4,
        Iterator[int],
    )  # pyright: ignore[reportUnusedCallResult]

    result5 = unique([1, 2, 3], exclude=[2])
    assert_type(
        result5,
        Iterator[int],
    )  # pyright: ignore[reportUnusedCallResult]

    # With key and include/exclude - filter values match key return type.
    result6 = unique(["a", "bb"], key=len, include=[1])
    assert_type(
        result6,
        Iterator[str],
    )  # pyright: ignore[reportUnusedCallResult]

    result7 = unique(["a", "bb"], key=len, exclude=[1])
    assert_type(
        result7,
        Iterator[str],
    )  # pyright: ignore[reportUnusedCallResult]
