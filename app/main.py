import logging
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.ratelimit import client_ip, limiter
from src.rag import answer, get_db, get_llm, retrieve, sources_from

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Load the embedding model and open the index now, not on the first question.
    get_db()
    try:
        get_llm()
    except RuntimeError as exc:
        logger.warning("LLM not configured: %s", exc)
    yield


app = FastAPI(
    title="Django Docs Q&A",
    description="Retrieval-augmented answers over the Django 5.2 documentation.",
    lifespan=lifespan,
)


def enforce_rate_limit(request: Request) -> None:
    host = request.client.host if request.client else None
    reason = limiter.check(client_ip(request.headers, host))
    if reason:
        raise HTTPException(status_code=429, detail=reason)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class Source(BaseModel):
    n: int
    source: str
    section: str
    url: str
    distance: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    refused: bool


class SearchHit(Source):
    content: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version() -> dict:
    # Render sets RENDER_GIT_COMMIT; the deploy workflow polls this to know the new release is live.
    return {"commit": os.getenv("RENDER_GIT_COMMIT", "dev")}


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(enforce_rate_limit)])
def ask(req: AskRequest) -> dict:
    try:
        return answer(req.question)
    except Exception:
        logger.exception("ask failed")
        raise HTTPException(status_code=502, detail="Could not generate an answer right now.")


@app.get("/search", response_model=list[SearchHit], dependencies=[Depends(enforce_rate_limit)])
def search(q: str = Query(min_length=3, max_length=500), k: int = Query(5, ge=1, le=10)) -> list[dict]:
    results = retrieve(q, k=k)
    hits = sources_from(results)
    for hit, (doc, _score) in zip(hits, results):
        hit["content"] = doc.page_content
    return hits

# Mount the Gradio chat UI at "/". This must come after the API routes above.
import gradio as gr  # noqa: E402
 
from app.ui import build_ui  # noqa: E402
 
app = gr.mount_gradio_app(app, build_ui(), path="/", footer_links=[])