from src import rag
from src.budget import GeminiBudget
from src.followup import normalize_history, recent_turns
from tests.test_rag import FakeLLM, make_results


def chat(*pairs):
    turns = []
    for question, reply in pairs:
        turns += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]
    return turns


def test_normalize_history_handles_gradio_shapes_and_strips_sources():
    history = [
        {"role": "user", "content": [{"type": "text", "text": "How do I cache a view?"}]},
        {"role": "assistant", "content": "Use cache_page [1].\n\n---\n**Sources**\n- [1] link"},
        {"role": "user", "content": ""},  # empty, dropped
        {"role": "tool", "content": "x"},  # unknown role, dropped
        "garbage",
    ]
    assert normalize_history(history) == [
        {"role": "user", "content": "How do I cache a view?"},
        {"role": "assistant", "content": "Use cache_page [1]."},
    ]


def test_recent_turns_keeps_last_three_exchanges_and_truncates():
    history = chat(*[(f"q{i}", "a" * 1000) for i in range(5)])
    turns = recent_turns(history)
    assert [t["content"] for t in turns if t["role"] == "user"] == ["q2", "q3", "q4"]
    assert all(len(t["content"]) <= 400 for t in turns)


def test_first_message_needs_no_rewrite_and_no_call(monkeypatch):
    monkeypatch.setattr(rag, "get_llm", lambda: (_ for _ in ()).throw(AssertionError("no LLM call expected")))
    assert rag.resolve_query("How do I cache a view?", [], mode="llm") == "How do I cache a view?"


def test_none_mode_ignores_history():
    history = chat(("How do I cache a view?", "Use cache_page."))
    assert rag.resolve_query("And with Redis?", history, mode="none") == "And with Redis?"


def test_llm_mode_uses_rewrite(monkeypatch):
    history = chat(("How do I cache a view?", "Use cache_page."))
    llm = FakeLLM("How do I use Redis as the cache backend?\nextra line ignored")
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    assert rag.resolve_query("And with Redis?", history, mode="llm") == "How do I use Redis as the cache backend?"
    assert llm.calls == 1


def test_llm_mode_uses_question_as_typed_when_budget_is_empty(monkeypatch):
    history = chat(("How do I cache a view?", "Use cache_page."))
    llm = FakeLLM("should not be called")
    monkeypatch.setattr(rag, "get_llm", lambda: llm)
    monkeypatch.setattr(rag, "budget", GeminiBudget(daily=0, per_minute=10))
    assert rag.resolve_query("And with Redis?", history, mode="llm") == "And with Redis?"
    assert llm.calls == 0


def test_llm_mode_uses_question_as_typed_when_rewrite_errors(monkeypatch):
    class Boom:
        def invoke(self, _m):
            raise RuntimeError("429")

    history = chat(("How do I cache a view?", "Use cache_page."))
    monkeypatch.setattr(rag, "get_llm", lambda: Boom())
    assert rag.resolve_query("And with Redis?", history, mode="llm") == "And with Redis?"
