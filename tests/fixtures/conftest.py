from __future__ import annotations

from collections.abc import Callable

import pytest
from faker import Faker


@pytest.fixture
def import_path_factory(fake: Faker) -> Callable[..., str]:
    """Factory: generates valid/invalid dotted import path strings."""

    def _create(valid: bool = True) -> str:
        if valid:
            return "os.path.join"
        first = fake.pystr(min_chars=4, max_chars=8)
        second = fake.pystr(min_chars=4, max_chars=8)
        return f"{first}.{second}"

    return _create
