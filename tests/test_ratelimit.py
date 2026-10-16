from fastapi.testclient import TestClient

from app import main, ui
from src.ratelimit import DAILY_LIMIT, TOO_FAST, RateLimiter, client_ip


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_blocks_after_per_minute_limit_then_recovers():
    clock = Clock()
    limiter = RateLimiter(per_minute=3, daily_cap=100, clock=clock)
    assert [limiter.check("a") for _ in range(3)] == [None, None, None]
    assert limiter.check("a") == TOO_FAST
    clock.now += 61
    assert limiter.check("a") is None


def test_clients_are_limited_independently():
    limiter = RateLimiter(per_minute=1, daily_cap=100, clock=Clock())
    assert limiter.check("a") is None
    assert limiter.check("b") is None
    assert limiter.check("a") == TOO_FAST


def test_daily_cap_is_per_client():
    clock = Clock()
    limiter = RateLimiter(per_minute=100, daily_cap=2, clock=clock)
    assert limiter.check("a") is None
    assert limiter.check("a") is None
    assert limiter.check("a") == DAILY_LIMIT
    assert limiter.check("b") is None  # another visitor is unaffected
    clock.now += 86401
    assert limiter.check("a") is None


def test_client_ip_prefers_first_forwarded_entry():
    assert client_ip({"x-forwarded-for": "1.2.3.4, 10.0.0.1"}, "127.0.0.1") == "1.2.3.4"
    assert client_ip({}, "127.0.0.1") == "127.0.0.1"
    assert client_ip({}, None) == "unknown"


def test_api_returns_429_when_limited(monkeypatch):
    monkeypatch.setattr(main, "limiter", RateLimiter(per_minute=1, daily_cap=100))
    monkeypatch.setattr(main, "answer", lambda q, history=None: {"answer": "ok", "sources": [], "refused": False})
    client = TestClient(main.app)
    body = {"question": "How do I create a custom user model?"}
    assert client.post("/ask", json=body).status_code == 200
    assert client.post("/ask", json=body).status_code == 429


def test_ui_returns_limit_message(monkeypatch):
    monkeypatch.setattr(ui, "limiter", RateLimiter(per_minute=1, daily_cap=100))
    monkeypatch.setattr(ui, "answer", lambda q, history=None: {"answer": "ok", "sources": [], "refused": False})
    assert ui.respond("How do I create a custom user model?", []) == "ok"
    assert ui.respond("How do I create a custom user model?", []) == TOO_FAST
