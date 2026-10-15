import threading
import time
from collections import defaultdict, deque

from src.config import DAILY_CAP, RATE_PER_MINUTE

TOO_FAST = "Too many questions, please wait a minute and try again."
DAILY_LIMIT = "The demo has reached its daily question limit. Please try again tomorrow."


class RateLimiter:
    """In-memory sliding-window limiter: N per minute per client, plus a daily cap overall.

    State lives in this process, so it resets on restart and is per instance.
    That is fine for a single-instance demo.
    """

    def __init__(self, per_minute: int, daily_cap: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._clock = clock
        self._lock = threading.Lock()
        self._per_client: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()

    def check(self, client: str) -> str | None:
        """Record a request. Returns None if allowed, or the message to show if not."""
        now = self._clock()
        with self._lock:
            while self._all and now - self._all[0] > 86400:
                self._all.popleft()
            if len(self._all) >= self.daily_cap:
                return DAILY_LIMIT

            recent = self._per_client[client]
            while recent and now - recent[0] > 60:
                recent.popleft()
            if len(recent) >= self.per_minute:
                return TOO_FAST

            recent.append(now)
            self._all.append(now)
            return None


def client_ip(headers, fallback: str | None) -> str:
    """First X-Forwarded-For entry (hosts put the real client there), else the socket address.

    A client can fake this header, which is why the daily cap exists as a backstop.
    """
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"


limiter = RateLimiter(RATE_PER_MINUTE, DAILY_CAP)
