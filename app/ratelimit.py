"""In-memory rate limits for LLM requests (single-process server).

Two limits protect a hosted LLM's quota on a public deployment:
- per client: LLM_REQUESTS_PER_MINUTE requests in any 60 seconds
- overall: LLM_REQUESTS_PER_DAY requests in any 24 hours, for everyone
  combined (can't be dodged by changing IP)
0 disables a limit.
"""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque

MINUTE, DAY = 60.0, 86400.0


class RateLimited(Exception):
    def __init__(self, message: str, retry_after: int):
        super().__init__(message)
        self.retry_after = retry_after


class LLMRateLimiter:
    def __init__(self, per_client_per_minute: int, per_day: int, clock=time.monotonic):
        self.per_client_per_minute = per_client_per_minute
        self.per_day = per_day
        self._clock = clock
        self._clients: dict[str, deque] = defaultdict(deque)
        self._all: deque = deque()
        self._lock = threading.Lock()

    def check(self, client: str) -> None:
        """Record a request from `client`, or raise RateLimited."""
        now = self._clock()
        with self._lock:
            recent = self._clients[client]
            _drop_older(recent, now - MINUTE)
            _drop_older(self._all, now - DAY)

            if self.per_day and len(self._all) >= self.per_day:
                raise RateLimited(
                    "The daily limit for AI simplification on this deployment has been "
                    "reached. Please try again later.",
                    retry_after=_wait(self._all[0] + DAY - now),
                )
            if self.per_client_per_minute and len(recent) >= self.per_client_per_minute:
                raise RateLimited(
                    "Too many AI requests. Please wait a minute and try again.",
                    retry_after=_wait(recent[0] + MINUTE - now),
                )
            recent.append(now)
            self._all.append(now)

            # Forget idle clients so the table doesn't grow without bound
            if len(self._clients) > 10_000:
                for key in [k for k, q in self._clients.items() if not q or q[-1] < now - MINUTE]:
                    del self._clients[key]


def _drop_older(q: deque, cutoff: float) -> None:
    while q and q[0] <= cutoff:
        q.popleft()


def _wait(seconds: float) -> int:
    return max(1, math.ceil(seconds))


def client_id(request) -> str:
    """Best-effort client address: the first X-Forwarded-For entry behind a
    proxy (as on Render). Clients can spoof that header, which is why there's
    also an overall daily limit."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
