import threading
import time
from collections import deque
from datetime import datetime
from zoneinfo import ZoneInfo

from src.config import GEMINI_DAILY_BUDGET, GEMINI_PER_MINUTE, QUOTA_TIMEZONE


class GeminiBudget:
    """Shared, in-memory budget for Gemini calls (all visitors together).

    Two limits: a daily total that resets at midnight in the quota timezone (when Google
    resets its own quota), and a per-minute sliding window. try_acquire() spends one call
    and returns False when either limit is used up.

    State lives in this process, so a restart (Render's free tier sleeps when idle) resets
    it. Gemini's own 429 errors are the real backstop; rag.answer handles those the same
    way as an empty budget.
    """

    def __init__(self, daily: int, per_minute: int, now=time.time, tz: str = QUOTA_TIMEZONE):
        self.daily = daily
        self.per_minute = per_minute
        self._now = now
        self._tz = ZoneInfo(tz)
        self._lock = threading.Lock()
        self._day = None
        self._used = 0
        self._recent: deque = deque()

    def _roll_day(self, now: float) -> None:
        day = datetime.fromtimestamp(now, self._tz).date()
        if day != self._day:
            self._day, self._used = day, 0

    def try_acquire(self) -> bool:
        now = self._now()
        with self._lock:
            self._roll_day(now)
            while self._recent and now - self._recent[0] > 60:
                self._recent.popleft()
            if self._used >= self.daily or len(self._recent) >= self.per_minute:
                return False
            self._used += 1
            self._recent.append(now)
            return True

    def remaining_today(self) -> int:
        with self._lock:
            self._roll_day(self._now())
            return max(self.daily - self._used, 0)


budget = GeminiBudget(GEMINI_DAILY_BUDGET, GEMINI_PER_MINUTE)
