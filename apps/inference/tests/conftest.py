import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(max_body_bytes=200_000, max_pending=2)
