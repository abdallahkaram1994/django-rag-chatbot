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
