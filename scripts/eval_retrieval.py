import json
import sys

from pathlib import Path

from langchain_chroma import Chroma

from src.embeddings import get_embedder

ROOT = Path(__file__).resolve().parent.parent
# Usage: python -m scripts.eval_retrieval [store_folder]   (default: chroma)
CHROMA_PATH = str(ROOT / (sys.argv[1] if len(sys.argv) > 1 else "chroma"))
QUESTIONS = ROOT / "evals" / "questions.jsonl"
K = 5


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def is_hit(doc, q: dict) -> bool:
    return any(
        doc.metadata["source"] == a["source"]
        and a["section_contains"].lower() in doc.metadata["section"].lower()
        for a in q["accept"]
    )


def first_hit_rank(results, q: dict) -> int | None:
    for rank, (doc, _score) in enumerate(results, start=1):
        if is_hit(doc, q):
            return rank
    return None


def main() -> None:
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedder())

    answerable, out_of_scope = [], []
    for q in load_questions(QUESTIONS):
        (answerable if q["accept"] else out_of_scope).append(q)

    ranks = []
    hit_dist, miss_dist = [], []
    for q in answerable:
        results = db.similarity_search_with_score(q["question"], k=K)
        rank = first_hit_rank(results, q)
        ranks.append((q, rank))
        (hit_dist if rank else miss_dist).append(results[0][1])
        if rank is None:
            print(f"\n[{q['id']}] {q['question']}")
            for doc, score in results[:3]:
                print(f"   {score:.3f}  {doc.metadata['source']} | {doc.metadata['section']}")

    n = len(ranks)
    for k in (1, 3, K):
        hits = sum(1 for _, r in ranks if r is not None and r <= k)
        print(f"hit@{k}: {hits}/{n} = {hits / n:.0%}")

    mrr = sum(1 / r for _, r in ranks if r is not None) / n
    print(f"MRR: {mrr:.3f}\n")

    print("Misses (not in top K):")
    for q, r in ranks:
        if r is None:
            print(f"  [{q['id']}] {q['question']}")

    print("\nBest distance, answerable questions (min / median / max):")
    for name, d in (("hits", hit_dist), ("misses", miss_dist)):
        if d:
            d = sorted(d)
            print(f"  {name}: {d[0]:.3f} / {d[len(d) // 2]:.3f} / {d[-1]:.3f}")

    print("\nOut-of-scope best distance (lower = closer):")
    for q in out_of_scope:
        _doc, score = db.similarity_search_with_score(q["question"], k=1)[0]
        print(f"  {score:.3f}  {q['question']}")


if __name__ == "__main__":
    sys.exit(main())
