import json
import logging
import re
import threading
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger("app.cache")


def normalize(query: str) -> str:
    """Lowercase, collapse whitespace, drop trailing punctuation, so trivial variants share an entry."""
    return re.sub(r"\s+", " ", query.strip().lower()).rstrip("?!. ")


class AnswerCache:
    """Answers keyed by the normalized query: a small LRU plus a read-only seed.

    The seed holds answers generated ahead of time (scripts/seed_cache.py) for the example
    questions, so demos of those cost no Gemini calls. It is never evicted. The LRU part is
    in memory only and starts empty after a restart.
    """

    def __init__(self, maxsize: int = 500):
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._lru: OrderedDict[str, dict] = OrderedDict()
        self._seed: dict[str, dict] = {}

    def get(self, query: str) -> dict | None:
        key = normalize(query)
        with self._lock:
            if key in self._lru:
                self._lru.move_to_end(key)
                return dict(self._lru[key])
            if key in self._seed:
                return dict(self._seed[key])
        return None

    def put(self, query: str, result: dict) -> None:
        key = normalize(query)
        with self._lock:
            self._lru[key] = dict(result)
            self._lru.move_to_end(key)
            while len(self._lru) > self.maxsize:
                self._lru.popitem(last=False)

    def load_seed(self, path: Path) -> int:
        """Load seeded answers if the file exists. Returns how many were loaded."""
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return 0
        except (OSError, ValueError):
            logger.exception("could not read cache seed %s", path)
            return 0
        with self._lock:
            for item in data.get("answers", []):
                self._seed[normalize(item["question"])] = item["result"]
            return len(self._seed)

    def __len__(self) -> int:
        with self._lock:
            return len(self._lru) + len(self._seed)
