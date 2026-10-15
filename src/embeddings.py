from langchain_core.embeddings import Embeddings

from src.config import EMBED_MODEL, MODEL_CACHE


class FastEmbedEmbeddings(Embeddings):
    """LangChain wrapper around fastembed (ONNX runtime, no torch).

    fastembed's bge models already return L2-normalized vectors.
    """

    def __init__(self, model_name: str = EMBED_MODEL, cache_dir: str = MODEL_CACHE):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(model_name=model_name, cache_dir=cache_dir)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts, batch_size=32)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([text]))).tolist()


def get_embedder() -> FastEmbedEmbeddings:
    return FastEmbedEmbeddings()
