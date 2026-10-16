import threading
import time
from collections import defaultdict, deque

from src.config import IP_DAILY_CAP, RATE_PER_MINUTE

TOO_FAST = "Too many questions, please wait a minute and try again."
DAILY_LIMIT = "You have reached the daily question limit for this demo. Please try again tomorrow."


class RateLimiter:
    """In-memory sliding-window limiter: N per minute and N per day, per client.

    This keeps one visitor from using up the shared Gemini budget (src/budget.py, which
    counts all visitors together). State lives in this process, so it resets on restart.
    """

    def __init__(self, per_minute: int, daily_cap: int, clock=time.monotonic):
        self.per_minute = per_minute
        self.daily_cap = daily_cap
        self._clock = clock
        self._lock = threading.Lock()
        self._per_client: dict[str, deque] = defaultdict(deque)

    def check(self, client: str) -> str | None:
        """Record a request. Returns None if allowed, or the message to show if not."""
        now = self._clock()
        with self._lock:
            history = self._per_client[client]
            while history and now - history[0] > 86400:
                history.popleft()
            if len(history) >= self.daily_cap:
                return DAILY_LIMIT
            if sum(1 for t in history if now - t <= 60) >= self.per_minute:
                return TOO_FAST

            history.append(now)
            if len(self._per_client) > 5000:  # forget clients with no requests in the last day
                self._per_client = defaultdict(
                    deque, {c: h for c, h in self._per_client.items() if h and now - h[-1] <= 86400}
                )
            return None


def client_ip(headers, fallback: str | None) -> str:
    """First X-Forwarded-For entry (hosts put the real client there), else the socket address.

    A client can fake this header, which is why the shared Gemini budget exists as a backstop.
    """
    forwarded = headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return fallback or "unknown"


limiter = RateLimiter(RATE_PER_MINUTE, IP_DAILY_CAP)
