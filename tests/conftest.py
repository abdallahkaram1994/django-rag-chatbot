import pytest

from app import main, ui
from src import rag
from src.budget import GeminiBudget
from src.cache import AnswerCache
from src.ratelimit import RateLimiter


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """Each test gets its own roomy limiter, Gemini budget and empty cache, so tests never affect each other."""
    limiter = RateLimiter(per_minute=1000, daily_cap=100000)
    monkeypatch.setattr(main, "limiter", limiter)
    monkeypatch.setattr(ui, "limiter", limiter)
    monkeypatch.setattr(rag, "budget", GeminiBudget(daily=100000, per_minute=100000))
    monkeypatch.setattr(rag, "cache", AnswerCache(100))
