import json
from datetime import datetime
from zoneinfo import ZoneInfo

from src.budget import GeminiBudget
from src.cache import AnswerCache, normalize


class Clock:
    def __init__(self, now: float):
        self.now = now

    def __call__(self):
        return self.now


def pacific(year, month, day, hour, minute=0) -> float:
    return datetime(year, month, day, hour, minute, tzinfo=ZoneInfo("America/Los_Angeles")).timestamp()


def test_daily_budget_runs_out_and_resets_at_pacific_midnight():
    clock = Clock(pacific(2026, 10, 15, 23, 50))
    budget = GeminiBudget(daily=2, per_minute=100, now=clock)
    assert budget.try_acquire() and budget.try_acquire()
    assert budget.try_acquire() is False
    assert budget.remaining_today() == 0
    clock.now = pacific(2026, 10, 16, 0, 5)  # next quota day
    assert budget.try_acquire() is True
    assert budget.remaining_today() == 1


def test_per_minute_window_slides():
    clock = Clock(pacific(2026, 10, 15, 12))
    budget = GeminiBudget(daily=100, per_minute=2, now=clock)
    assert budget.try_acquire() and budget.try_acquire()
    assert budget.try_acquire() is False
    clock.now += 61
    assert budget.try_acquire() is True


def test_normalize_merges_trivial_variants():
    assert normalize("  How do I   cache a view?? ") == normalize("how do i cache a view")


def test_cache_get_put_and_lru_eviction():
    cache = AnswerCache(maxsize=2)
    cache.put("a one", {"answer": "1"})
    cache.put("b two", {"answer": "2"})
    assert cache.get("A one?")["answer"] == "1"  # also refreshes "a one"
    cache.put("c three", {"answer": "3"})  # evicts "b two", the least recently used
    assert cache.get("b two") is None
    assert cache.get("c three")["answer"] == "3"


def test_cache_returns_copies():
    cache = AnswerCache(5)
    cache.put("q one", {"answer": "x"})
    cache.get("q one")["answer"] = "changed"
    assert cache.get("q one")["answer"] == "x"


def test_seed_is_loaded_and_never_evicted(tmp_path):
    seed = tmp_path / "seed.json"
    seed.write_text(json.dumps({"answers": [{"question": "Seeded question?", "result": {"answer": "S"}}]}))
    cache = AnswerCache(maxsize=1)
    assert cache.load_seed(seed) == 1
    cache.put("x one", {"answer": "1"})
    cache.put("y two", {"answer": "2"})
    assert cache.get("seeded question")["answer"] == "S"


def test_missing_or_broken_seed_is_ignored(tmp_path):
    cache = AnswerCache(5)
    assert cache.load_seed(tmp_path / "nope.json") == 0
    broken = tmp_path / "broken.json"
    broken.write_text("{not json")
    assert cache.load_seed(broken) == 0
