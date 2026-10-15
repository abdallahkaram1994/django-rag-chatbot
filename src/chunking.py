import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.preprocess import clean_chunk

HEADING_SEP = r"\n(?=[^\n]+\n[=\-~^]{3,}\n)"
DIRECTIVE_SEP = r"\n(?=[ \t]*\.\. [\w-]+::)"

HEADING_RE = re.compile(r"^(?:([=\-~^])\1{2,}\n)?([^\n]+)\n([=\-~^])\3{2,}\n", re.MULTILINE)


def heading_trail(text: str, position: int) -> str:
    levels = {}   # underline style -> depth, in order of first appearance
    stack = []    # current trail of titles

    for m in HEADING_RE.finditer(text):
        if m.start() > position:
            break
        style = (m.group(3), m.group(1) is not None)   # (underline char, has overline?)
        level = levels.setdefault(style, len(levels))
        stack = stack[:level] + [m.group(2).strip()]

    if not stack:
        # chunk starts before the first heading (e.g. a ".. _label:" line above the title)
        first = HEADING_RE.search(text)
        return first.group(2).strip() if first else ""

    return " > ".join(stack)


def _same_section(a: Document, b: Document) -> bool:
    return (
        a.metadata["source"] == b.metadata["source"]
        and a.metadata["section"] == b.metadata["section"]
    )


def merge_small(chunks: list[Document], min_chars: int = 150) -> list[Document]:
    # Pass 1: a small chunk is prepended to the next chunk if they share source + section
    forward = []
    pending = None
    for c in chunks:
        if pending is not None:
            if _same_section(pending, c):
                c.page_content = pending.page_content + "\n\n" + c.page_content
                c.metadata["start_index"] = pending.metadata["start_index"]
            else:
                forward.append(pending)
            pending = None
        if len(c.page_content) < min_chars:
            pending = c
        else:
            forward.append(c)
    if pending is not None:
        forward.append(pending)

    # Pass 2: any small chunk left over is appended to the previous chunk if they share source + section
    merged = []
    for c in forward:
        if merged and len(c.page_content) < min_chars and _same_section(merged[-1], c):
            merged[-1].page_content += "\n\n" + c.page_content
        else:
            merged.append(c)
    return merged


def chunk_docs(docs, chunk_size, chunk_overlap, min_chars=150):
    originals = {d.metadata["source"]: d.page_content for d in docs}

    splitter = RecursiveCharacterTextSplitter(
        separators=[HEADING_SEP, DIRECTIVE_SEP, r"(?<!::)\n\n", "\n", " ", ""],
        is_separator_regex=True,
        keep_separator=True,
        strip_whitespace=False,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)

    for c in chunks:
        original = originals[c.metadata["source"]]
        text = c.page_content

        leading = len(text) - len(text.lstrip())
        position = c.metadata["start_index"] + leading

        c.metadata["section"] = heading_trail(original, position)
        c.page_content = clean_chunk(text)

    chunks = [c for c in chunks if c.page_content.strip()]
    return merge_small(chunks, min_chars)
