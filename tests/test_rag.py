from langchain_core.documents import Document

from src import rag


def make_results(distances):
    return [
        (
            Document(
                page_content=f"content {i}",
                metadata={"source": "topics/db/queries.txt", "section": f"Making queries > S{i}"},
            ),
            d,
        )
        for i, d in enumerate(distances)
    ]


class FakeMessage:
    def __init__(self, text):
        self.text = text


class FakeLLM:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def invoke(self, _messages):
        self.calls += 1
        return FakeMessage(self.text)


def test_doc_url():
    assert rag.doc_url("topics/auth/customizing.txt") == (
        "https://docs.djangoproject.com/en/5.2/topics/auth/customizing/"
    )
    assert rag.doc_url("ref/models/index.txt") == "https://docs.djangoproject.com/en/5.2/ref/models/"


def test_build_context_numbers_chunks():
    context = rag.build_context(make_results([0.3, 0.4]))
    assert context.startswith("[1] (Making queries > S0)")
    assert "[2] (Making queries > S1)" in context


def test_far_match_refuses_without_calling_llm(monkeypatch):
    llm = FakeLLM("should not be used")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.9]))
    monkeypatch.setattr(rag, "get_llm", lambda: llm)

    result = rag.answer("What is the capital of France?")

    assert result["refused"] is True
    assert result["sources"] == []
    assert llm.calls == 0


def test_close_match_returns_answer_with_sources(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.3, 0.4]))
    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM("Use AbstractUser [1]."))

    result = rag.answer("How do I create a custom user model?")

    assert result["refused"] is False
    assert result["answer"] == "Use AbstractUser [1]."
    assert [s["n"] for s in result["sources"]] == [1, 2]


def test_model_says_not_covered(monkeypatch):
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.5]))
    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM("NOT_COVERED"))

    result = rag.answer("Something borderline")

    assert result["refused"] is True
    assert result["answer"] == rag.NOT_COVERED


def test_repeat_question_is_served_from_cache(monkeypatch):
    llm = FakeLLM("Use AbstractUser [1].")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.3, 0.4]))
    monkeypatch.setattr(rag, "get_llm", lambda: llm)

    first = rag.answer("How do I create a custom user model?")
    second = rag.answer("  how do i create a custom user model ")

    assert llm.calls == 1
    assert second["answer"] == first["answer"]
    assert second["degraded"] is False


def test_empty_budget_returns_sources_without_calling_llm(monkeypatch):
    from src.budget import GeminiBudget

    llm = FakeLLM("unused")
    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.3, 0.4, 0.5, 0.55]))
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    monkeypatch.setattr(rag, "budget", GeminiBudget(daily=0, per_minute=10))

    result = rag.answer("How do I create a custom user model?")

    assert result["degraded"] is True
    assert result["refused"] is False
    assert [s["n"] for s in result["sources"]] == [1, 2, 3]
    assert llm.calls == 0
    assert rag.cache.get("How do I create a custom user model?") is None  # degraded results are not cached


def test_gemini_error_falls_back_to_sources(monkeypatch):
    class Boom:
        def invoke(self, _m):
            raise RuntimeError("429 quota exceeded")

    monkeypatch.setattr(rag, "retrieve", lambda q, k=5: make_results([0.3]))
    monkeypatch.setattr(rag, "get_llm", lambda: Boom())

    result = rag.answer("How do I create a custom user model?")

    assert result["degraded"] is True
    assert result["sources"]


def test_result_reports_the_query_used(monkeypatch):
    seen = {}

    def retrieve(q, k=5):
        seen.setdefault("queries", []).append(q)
        return make_results([0.3])

    monkeypatch.setattr(rag, "retrieve", retrieve)
    monkeypatch.setattr(rag, "get_llm", lambda: FakeLLM("ok [1]"))
    history = [
        {"role": "user", "content": "How do I cache a view?"},
        {"role": "assistant", "content": "Use cache_page."},
    ]
    result = rag.answer("And with Redis?", history, mode="none")
    assert result["query"] == "And with Redis?"
