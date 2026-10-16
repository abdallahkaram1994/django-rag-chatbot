import logging
from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.budget import budget
from src.cache import AnswerCache
from src.config import CACHE_SIZE, CHROMA_PATH, FOLLOWUP_MODE, GEMINI_MODEL, MAX_DISTANCE, TOP_K
from src.embeddings import get_embedder
from src.followup import REWRITE_PROMPT, format_history, recent_turns

logger = logging.getLogger("app.rag")

cache = AnswerCache(CACHE_SIZE)

DOCS_BASE = "https://docs.djangoproject.com/en/5.2/"

NOT_COVERED = (
    "I couldn't find this in the Django 5.2 documentation. "
    "Try rephrasing, or ask about a specific Django feature."
)

DEGRADED_NOTICE = (
    "I can't generate an answer right now (this free demo has a small shared AI quota), "
    "so here are the most relevant documentation sections instead:"
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a documentation assistant for Django 5.2. Answer the user's question "
            "using the numbered documentation excerpts below.\n\n"
            "Rules:\n"
            "- Base every statement on the excerpts. Do not add facts from outside them.\n"
            "- Cite the excerpts you used inline, like [1] or [2][3].\n"
            "- If the question is broad (for example, how to use authentication), give a short "
            "overview of what the excerpts cover and point to the most relevant ones, "
            "instead of refusing.\n"
            "- If the excerpts only partly answer the question, answer the part they cover "
            "and say what they do not cover.\n"
            "- Reply with exactly NOT_COVERED only if the excerpts are unrelated to the question.\n"
            "- Put code, template tags, HTML tags and setting names inside backticks or fenced code blocks.\n"
            "- Be concise. Include short code examples from the excerpts when they help.\n\n"
            "Excerpts:\n{context}",
        ),
        ("human", "{question}"),
    ]
)


@lru_cache(maxsize=1)
def get_db() -> Chroma:
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedder())


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    if not GEMINI_MODEL:
        raise RuntimeError("Set GEMINI_MODEL (and GOOGLE_API_KEY) in .env")
    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=0)


def retrieve(question: str, k: int = TOP_K) -> list[tuple[Document, float]]:
    return get_db().similarity_search_with_score(question, k=k)


def doc_url(source: str) -> str:
    """topics/auth/customizing.txt -> https://docs.djangoproject.com/en/5.2/topics/auth/customizing/"""
    path = source.removesuffix(".txt")
    path = path.removesuffix("index").rstrip("/")
    return DOCS_BASE + (path + "/" if path else "")


def build_context(results: list[tuple[Document, float]]) -> str:
    blocks = []
    for i, (doc, _score) in enumerate(results, start=1):
        blocks.append(f"[{i}] ({doc.metadata['section']})\n{doc.page_content}")
    return "\n\n".join(blocks)


def sources_from(results: list[tuple[Document, float]]) -> list[dict]:
    return [
        {
            "n": i,
            "source": doc.metadata["source"],
            "section": doc.metadata["section"],
            "url": doc_url(doc.metadata["source"]),
            "distance": round(score, 3),
        }
        for i, (doc, score) in enumerate(results, start=1)
    ]


def rewrite_followup(question: str, turns: list[dict]) -> str | None:
    """Ask Gemini for a standalone version of a follow-up. None if it fails or looks wrong."""
    if not budget.try_acquire():
        return None
    try:
        messages = REWRITE_PROMPT.format_messages(history=format_history(turns), question=question)
        text = get_llm().invoke(messages).text.strip()
    except Exception:
        logger.exception("follow-up rewrite failed")
        return None
    line = text.splitlines()[0].strip() if text else ""
    return line if 3 <= len(line) <= 500 else None


def pick_query(question: str, previous: str) -> str:
    """Free follow-up handling: use whichever of 'question' or 'previous + question' retrieves closer."""
    joined = f"{previous} {question}"
    alone = retrieve(question, k=1)
    together = retrieve(joined, k=1)
    if alone and together and together[0][1] < alone[0][1]:
        return joined
    return question


def resolve_query(question: str, history=None, mode: str | None = None) -> str:
    """Turn the user's message into the query used for retrieval and generation."""
    mode = mode or FOLLOWUP_MODE
    turns = recent_turns(history)
    previous_questions = [t["content"] for t in turns if t["role"] == "user"]
    if mode == "none" or not previous_questions:
        return question  # first message: nothing to resolve, no extra call
    if mode == "llm":
        rewritten = rewrite_followup(question, turns)
        if rewritten:
            return rewritten
    return pick_query(question, previous_questions[-1])


def retrieval_only(results: list[tuple[Document, float]], query: str) -> dict:
    """Used when no Gemini call is available: show the best-matching sections instead of an answer."""
    return {
        "answer": DEGRADED_NOTICE,
        "sources": sources_from(results[:3]),
        "refused": False,
        "degraded": True,
        "query": query,
    }


def answer(question: str, history=None, mode: str | None = None) -> dict:
    query = resolve_query(question, history, mode)

    hit = cache.get(query)
    if hit is not None:
        return {**hit, "degraded": False, "query": query}

    results = retrieve(query)
    if not results or results[0][1] > MAX_DISTANCE:
        return {"answer": NOT_COVERED, "sources": [], "refused": True, "degraded": False, "query": query}

    if not budget.try_acquire():
        return retrieval_only(results, query)
    try:
        messages = PROMPT.format_messages(context=build_context(results), question=query)
        text = get_llm().invoke(messages).text.strip()
    except Exception:
        # Quota exhausted on Google's side, network trouble, or a missing key: still show something useful.
        logger.exception("generation failed")
        return retrieval_only(results, query)

    if text.startswith("NOT_COVERED"):
        result = {"answer": NOT_COVERED, "sources": [], "refused": True}
    else:
        result = {"answer": text, "sources": sources_from(results), "refused": False}
    cache.put(query, result)
    return {**result, "degraded": False, "query": query}
