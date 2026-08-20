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


@pytest.fixture
def monkeypatch_target_factory(fake: Faker) -> Callable[..., object]:
    """Factory: creates a simple object with patchable attributes."""

    def _create(**overrides: object) -> object:
        class Target:
            attr1 = fake.pystr(min_chars=4, max_chars=8)
            attr2 = fake.pystr(min_chars=4, max_chars=8)

        for k, v in overrides.items():
            setattr(Target, k, v)
        return Target()

    return _create
