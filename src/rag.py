from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import CHROMA_PATH, GEMINI_MODEL, MAX_DISTANCE, TOP_K
from src.embeddings import get_embedder

DOCS_BASE = "https://docs.djangoproject.com/en/5.2/"

NOT_COVERED = (
    "I couldn't find this in the Django 5.2 documentation. "
    "Try rephrasing, or ask about a specific Django feature."
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


def answer(question: str) -> dict:
    results = retrieve(question)
    if not results or results[0][1] > MAX_DISTANCE:
        return {"answer": NOT_COVERED, "sources": [], "refused": True}

    messages = PROMPT.format_messages(context=build_context(results), question=question)
    text = get_llm().invoke(messages).text.strip()

    if text.startswith("NOT_COVERED"):
        return {"answer": NOT_COVERED, "sources": [], "refused": True}
    return {"answer": text, "sources": sources_from(results), "refused": False}
