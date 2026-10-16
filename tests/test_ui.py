from app import ui


def src(n, section, url="https://docs.djangoproject.com/en/5.2/topics/db/queries/"):
    return {"n": n, "section": section, "url": url, "distance": 0.4}


def test_same_section_is_merged():
    sources = [src(1, "Queries > Q objects"), src(2, "Queries > Other"), src(3, "Queries > Q objects")]
    groups = ui.group_sources(sources, "See [1] and [3] and [2].")
    assert groups == [
        ("Queries > Q objects", sources[0]["url"] + "#q-objects", [1, 3]),
        ("Queries > Other", sources[0]["url"] + "#other", [2]),
    ]


def test_only_cited_sources_are_shown():
    sources = [src(1, "A > B"), src(2, "A > C"), src(3, "A > D")]
    groups = ui.group_sources(sources, "Use this [2].")
    assert [ns for _, _, ns in groups] == [[2]]


def test_no_citations_shows_everything():
    sources = [src(1, "A > B"), src(2, "A > C")]
    assert len(ui.group_sources(sources, "No citations here.")) == 2


def test_format_reply_lists_merged_numbers():
    result = {
        "answer": "Do X [1][3].",
        "refused": False,
        "sources": [src(1, "A > B"), src(2, "A > C"), src(3, "A > B")],
    }
    reply = ui.format_reply(result)
    assert "**[1, 3]** [A > B]" in reply
    assert "A > C" not in reply


def test_anchor_follows_sphinx_heading_ids():
    assert ui.anchor("Making queries > Complex lookups with ``Q`` objects") == "#complex-lookups-with-q-objects"
    assert ui.anchor("QuerySet API > Methods > ``select_related()``") == "#select-related"
    assert ui.anchor("Deployment checklist > Critical settings > :setting:`DEBUG`") == "#debug"
    assert ui.anchor("Pagination") == ""  # page title: no anchor


def test_indented_open_unindented_close_does_not_swallow_the_rest():
    bad = "- item:\n  ```python\n  x = 1\n```\n\n### After\ntext"
    fixed = ui.normalize_fences(bad)
    assert fixed == "- item:\n```python\nx = 1\n```\n\n### After\ntext"
    assert fixed.count("```") % 2 == 0


def test_unclosed_fence_is_closed():
    assert ui.normalize_fences("a\n```python\nx = 1") == "a\n```python\nx = 1\n```"


def test_well_formed_text_is_unchanged():
    ok = "text\n\n```python\nx = 1\n```\n\nmore"
    assert ui.normalize_fences(ok) == ok


def test_respond_passes_history_to_answer(monkeypatch):
    seen = {}

    def fake(message, history=None):
        seen["history"] = history
        return {"answer": "ok", "sources": [], "refused": False}

    monkeypatch.setattr(ui, "answer", fake)
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    assert ui.respond("a follow-up question", history) == "ok"
    assert seen["history"] == history
