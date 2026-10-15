import pytest

from app import main, ui
from src.ratelimit import RateLimiter


@pytest.fixture(autouse=True)
def fresh_limiter(monkeypatch):
    """Each test gets its own roomy limiter so tests never affect each other."""
    limiter = RateLimiter(per_minute=1000, daily_cap=100000)
    monkeypatch.setattr(main, "limiter", limiter)
    monkeypatch.setattr(ui, "limiter", limiter)
