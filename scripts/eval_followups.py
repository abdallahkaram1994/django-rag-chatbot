"""Compare follow-up modes on evals/followups.jsonl.

Usage (from the project root):
    python -m scripts.eval_followups            # none + llm
    python -m scripts.eval_followups none       # free, local only; llm uses one Gemini call per pair

For answerable follow-ups it reports hit@1/3/5 and MRR on the retrieved sections. For the
out-of-scope ones it reports whether the distance gate would still refuse them.
"""

import json
import sys

from scripts.eval_retrieval import K, first_hit_rank
from src import rag
from src.config import MAX_DISTANCE, ROOT

PATH = ROOT / "evals" / "followups.jsonl"


def load() -> list[dict]:
    with open(PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def run(mode: str, rows: list[dict]) -> None:
    ranks, refused_ok, misses = [], [], []
    for row in rows:
        query = rag.resolve_query(row["question"], row["history"], mode)
        results = rag.retrieve(query, k=K)
        if row["accept"]:
            rank = first_hit_rank(results, row)
            ranks.append(rank)
            if rank is None:
                misses.append((row, query, results[0][0].metadata["section"]))
        else:
            refused_ok.append(results[0][1] > MAX_DISTANCE)
    n = len(ranks)
    parts = [f"hit@{k}: {sum(1 for r in ranks if r and r <= k)}/{n}" for k in (1, 3, K)]
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{mode:7s} " + "  ".join(parts) + f"  MRR {mrr:.3f}  out-of-scope refused: {sum(refused_ok)}/{len(refused_ok)}")
    for row, query, top in misses:
        print(f"    miss [{row['id']}] {row['question']!r}\n         query used: {query!r}\n         top result: {top}")


def main() -> None:
    modes = (sys.argv[1] if len(sys.argv) > 1 else "none,llm").split(",")
    rows = load()
    for mode in modes:
        run(mode, rows)


if __name__ == "__main__":
    main()
