from pathlib import Path
from langchain_core.documents import Document

def load_docs(path: str | Path) -> list[Document]:
    root = Path(path)
    files = sorted(root.rglob("*.txt"))
    docs = []

    for file in files:
        text = file.read_text(encoding="utf-8")
        if not text.strip():
            continue
        rel = file.relative_to(root).as_posix()
        
        docs.append(
            Document(
                page_content=text,
                metadata={"source": rel}
            )
        )

    return docs