from fastapi.testclient import TestClient

from app import main

client = TestClient(main.app)

FAKE_ANSWER = {
    "answer": "Use AbstractUser [1].",
    "sources": [
        {
            "n": 1,
            "source": "topics/auth/customizing.txt",
            "section": "A > B",
            "url": "https://docs.djangoproject.com/en/5.2/topics/auth/customizing/",
            "distance": 0.39,
        }
    ],
    "refused": False,
}


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_version_reports_commit(monkeypatch):
    monkeypatch.setenv("RENDER_GIT_COMMIT", "abc123")
    assert client.get("/version").json() == {"commit": "abc123"}


def test_ask_ok(monkeypatch):
    monkeypatch.setattr(main, "answer", lambda q: FAKE_ANSWER)
    r = client.post("/ask", json={"question": "How do I create a custom user model?"})
    assert r.status_code == 200
    assert r.json()["sources"][0]["n"] == 1


def test_ask_rejects_short_and_long_questions():
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "x" * 501}).status_code == 422
    assert client.post("/ask", json={}).status_code == 422


def test_ask_returns_502_when_generation_fails(monkeypatch):
    def boom(_q):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(main, "answer", boom)
    r = client.post("/ask", json={"question": "How do I create a custom user model?"})
    assert r.status_code == 502
    assert "quota" not in r.text  # internal error details are not leaked
