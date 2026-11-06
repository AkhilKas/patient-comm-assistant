"""LLM rate limits, with a fake clock."""

from types import SimpleNamespace

import pytest

from app.ratelimit import LLMRateLimiter, RateLimited, client_id


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_per_client_limit_and_reset():
    clock = Clock()
    limiter = LLMRateLimiter(per_client_per_minute=2, per_day=0, clock=clock)
    limiter.check("a")
    limiter.check("a")
    with pytest.raises(RateLimited) as exc:
        limiter.check("a")
    assert 1 <= exc.value.retry_after <= 60
    limiter.check("b")  # other clients are unaffected

    clock.now += 61
    limiter.check("a")


def test_daily_limit_covers_all_clients():
    clock = Clock()
    limiter = LLMRateLimiter(per_client_per_minute=0, per_day=3, clock=clock)
    for client in ("a", "b", "c"):
        limiter.check(client)
    with pytest.raises(RateLimited, match="daily limit"):
        limiter.check("d")
    clock.now += 86401
    limiter.check("d")


def test_refused_requests_are_not_counted():
    clock = Clock()
    limiter = LLMRateLimiter(per_client_per_minute=1, per_day=2, clock=clock)
    limiter.check("a")
    for _ in range(5):
        with pytest.raises(RateLimited):
            limiter.check("a")
    limiter.check("b")  # still room in the daily budget


def test_zero_disables_limits():
    limiter = LLMRateLimiter(per_client_per_minute=0, per_day=0)
    for _ in range(100):
        limiter.check("a")


def test_client_id_prefers_forwarded_for():
    req = SimpleNamespace(headers={"x-forwarded-for": "203.0.113.7, 10.0.0.1"},
                          client=SimpleNamespace(host="10.0.0.1"))
    assert client_id(req) == "203.0.113.7"
    req = SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1"))
    assert client_id(req) == "127.0.0.1"
