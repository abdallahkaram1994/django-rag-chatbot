"""Build the Chroma index from the docs.

Usage (from the project root):
    python -m scripts.build_index                      # builds ./chroma
    python -m scripts.build_index chroma_prefix prefix # section title added to the embedded text
"""
import shutil
import sys
from pathlib import Path

from langchain_chroma import Chroma

from src.chunking import chunk_docs
from src.embeddings import get_embedder
from src.loading import load_docs

ROOT = Path(__file__).resolve().parent.parent
BATCH = 1000  # Chroma rejects very large single inserts


def main(folder: str, with_section: bool) -> None:
    store = ROOT / folder
    if store.exists():
        shutil.rmtree(store)  # rebuild from scratch so reruns never duplicate chunks

    chunks = chunk_docs(load_docs(str(ROOT / "docs")), 1000, 100)
    if with_section:
        for c in chunks:
            c.page_content = f"{c.metadata['section']}\n\n{c.page_content}"

    db = Chroma(persist_directory=str(store), embedding_function=get_embedder())
    for i in range(0, len(chunks), BATCH):
        db.add_documents(chunks[i : i + BATCH])
        print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}", flush=True)
    print(f"Built {len(chunks)} chunks -> {store}")


if __name__ == "__main__":
    main(
        sys.argv[1] if len(sys.argv) > 1 else "chroma",
        with_section=len(sys.argv) > 2 and sys.argv[2] == "prefix",
    )
