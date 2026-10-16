"""Helpers for follow-up questions: clean up chat history and build the rewrite prompt."""

from langchain_core.prompts import ChatPromptTemplate

from src.config import HISTORY_TURNS

SOURCES_MARKER = "\n\n---\n**Sources**"  # appended by the UI to every reply
MAX_USER_CHARS = 300
MAX_ASSISTANT_CHARS = 400

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite the latest message in a chat about the Django web framework as one "
            "standalone question that makes sense without the conversation.\n"
            "- Replace pronouns and references like 'that', 'it', 'those' with what they refer to.\n"
            "- If the message is already standalone, or changes to a new topic, return it unchanged.\n"
            "- Do not answer the question. Output only the question, on one line.",
        ),
        ("human", "Conversation so far:\n{history}\n\nLatest message: {question}"),
    ]
)


def _text(content) -> str:
    """Chat content as plain text. Gradio can hand back a list of parts instead of a string."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return " ".join(_text(part) for part in content)
    return ""


def normalize_history(history) -> list[dict]:
    """[{'role': 'user'|'assistant', 'content': str}], dropping anything malformed or empty."""
    turns = []
    for item in history or []:
        role = item.get("role") if isinstance(item, dict) else getattr(item, "role", None)
        content = item.get("content") if isinstance(item, dict) else getattr(item, "content", None)
        text = _text(content).split(SOURCES_MARKER)[0].strip()
        if role in ("user", "assistant") and text:
            turns.append({"role": role, "content": text})
    return turns


def recent_turns(history, exchanges: int = HISTORY_TURNS) -> list[dict]:
    """The last few exchanges, each message shortened."""
    turns = normalize_history(history)[-2 * exchanges :]
    limits = {"user": MAX_USER_CHARS, "assistant": MAX_ASSISTANT_CHARS}
    return [{"role": t["role"], "content": t["content"][: limits[t["role"]]]} for t in turns]


def format_history(turns: list[dict]) -> str:
    names = {"user": "User", "assistant": "Assistant"}
    return "\n".join(f"{names[t['role']]}: {t['content']}" for t in turns)
