import logging
import re
import unicodedata

import gradio as gr

from src.ratelimit import client_ip, limiter
from src.rag import answer

logger = logging.getLogger("app.ui")

EXAMPLES = [
    "How do I create a custom user model?",
    "How do I write a data migration?",
    "How do I cache the output of a single view?",
    "What settings should I change before deploying to production?",
]

MAX_QUESTION_CHARS = 500


def clean_label(section: str) -> str:
    """Make an RST section trail safe to use as markdown link text."""
    section = re.sub(r":[\w:+-]+:`([^`]*)`", r"\1", section)  # :setting:`DEBUG` -> DEBUG
    section = section.replace("`", "").replace("\\", "")  # backticks, "\s" leftovers
    section = re.sub(r"[\[\]<*|]", "", section)  # characters that break links or emphasis
    return section.strip()


def anchor(section: str) -> str:
    """Sphinx heading id for the last heading in a section trail, e.g. '#select-related'.

    Follows docutils' id rules: lowercase, runs of other characters become '-',
    no leading digits or hyphens, no trailing hyphens. A one-level trail is the
    page title, so it gets no anchor.
    """
    parts = section.split(" > ")
    if len(parts) < 2:
        return ""
    title = unicodedata.normalize("NFKD", clean_label(parts[-1])).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower())
    slug = re.sub(r"^[-0-9]+|-+$", "", slug)
    return f"#{slug}" if slug else ""


def normalize_fences(text: str) -> str:
    """Make code fences well-formed so one bad fence can't swallow the rest of the reply.

    Models sometimes indent an opening fence (inside a list item) but not the closing one.
    Markdown then ends the list at the closing fence and treats it as a NEW opening fence,
    which turns everything after it into code. We put every fence at column 0, remove the
    opening fence's indent from the code lines, and close a fence the model left open.
    """
    out: list[str] = []
    in_code = False
    indent = 0
    for line in text.split("\n"):
        m = re.match(r"^(\s*)(`{3,})(.*)$", line)
        if m:
            if not in_code:
                in_code, indent = True, len(m.group(1))
                out.append(m.group(2) + m.group(3).strip())
            else:
                in_code = False
                out.append(m.group(2))
            continue
        if in_code and indent and line.startswith(" " * indent):
            line = line[indent:]
        out.append(line)
    if in_code:
        out.append("```")
    return "\n".join(out)


def escape_tags(text: str) -> str:
    """Show stray <tags> as text. Code blocks and `inline code` are left alone."""
    parts = re.split(r"(```.*?```|`[^`\n]*`)", text, flags=re.DOTALL)
    return "".join(p if i % 2 else p.replace("<", "&lt;") for i, p in enumerate(parts))


def group_sources(sources: list[dict], answer_text: str) -> list[tuple[str, str, list[int]]]:
    """Keep the excerpts the answer cites, and merge ones from the same section.

    Returns (label, url, [excerpt numbers]) tuples. If the answer cites nothing,
    all sources are kept.
    """
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer_text)}
    chosen = [s for s in sources if s["n"] in cited] or sources
    groups: dict[tuple[str, str], list[int]] = {}
    for s in chosen:
        groups.setdefault((clean_label(s["section"]), s["url"] + anchor(s["section"])), []).append(s["n"])
    return [(label, url, ns) for (label, url), ns in groups.items()]


def format_reply(result: dict) -> str:
    text = escape_tags(normalize_fences((result.get("answer") or "").strip()))
    if not text:
        text = "The model returned an empty answer. Please try again."
    if result["sources"]:
        lines = [
            f"- **[{', '.join(map(str, ns))}]** [{label}]({url})"
            for label, url, ns in group_sources(result["sources"], result["answer"])
        ]
        text += "\n\n---\n**Sources**\n" + "\n".join(lines)
    return text


def respond(message: str, history: list, request: gr.Request | None = None) -> str:
    message = message.strip()
    if len(message) < 3:
        return "Please ask a longer question."
    if len(message) > MAX_QUESTION_CHARS:
        return f"Please keep questions under {MAX_QUESTION_CHARS} characters."
    host = request.client.host if request is not None and request.client else None
    headers = request.headers if request is not None else {}
    reason = limiter.check(client_ip(headers, host))
    if reason:
        return reason
    try:
        return format_reply(answer(message))
    except Exception:
        logger.exception("respond failed")
        return "Something went wrong generating an answer. Please try again in a moment."


def build_ui() -> gr.ChatInterface:
    return gr.ChatInterface(
        fn=respond,
        chatbot=gr.Chatbot(
            latex_delimiters=[],  # no math rendering: "$" and "\\(" in Django text must stay literal
            sanitize_html=True,
            height="calc(100vh - 250px)",
        ),
        title="Django Docs Q&A",
        description=(
            "Ask a question about Django 5.2. Answers are generated from the official "
            "documentation and cite their sources. Each question is answered on its own; "
            "earlier messages are not used as context."
        ),
        examples=EXAMPLES,
        cache_examples=False,
    )
