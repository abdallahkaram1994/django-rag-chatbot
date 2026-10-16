"""Generate answers for the example questions once and save them, so demos of them cost no Gemini calls.

Usage (from the project root):  python -m scripts.seed_cache
Uses one Gemini call per example question. Re-run after changing the index, the prompt or EXAMPLES,
then commit cache/seed.json.
"""

import json

from app.ui import EXAMPLES
from src.config import CACHE_SEED
from src.rag import answer


def main() -> None:
    items = []
    for question in EXAMPLES:
        result = answer(question, mode="none")
        if result["degraded"]:
            raise SystemExit(f"No answer generated for {question!r}; check GEMINI_MODEL and your quota.")
        items.append(
            {"question": question, "result": {k: result[k] for k in ("answer", "sources", "refused")}}
        )
        print("seeded:", question)
    CACHE_SEED.parent.mkdir(exist_ok=True)
    CACHE_SEED.write_text(json.dumps({"answers": items}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(items)} answers to {CACHE_SEED}")


if __name__ == "__main__":
    main()
