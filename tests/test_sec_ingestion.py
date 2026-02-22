from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import pytest

from src.ingest import fundamentals_sec as sec


class _FakeResp:
    def __init__(self, status_code: int, payload: dict[str, Any] | None = None):
        self.status_code = status_code
        self._payload = payload or {}
        self.content = json.dumps(self._payload).encode("utf-8")

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def test_rate_limiter_enforces_limit() -> None:
    limiter = sec.TokenBucketRateLimiter(rate=9.0, capacity=1.0)

    async def _run() -> list[float]:
        stamps: list[float] = []
        for _ in range(50):
            await limiter.acquire()
            stamps.append(time.monotonic())
        return stamps

    stamps = asyncio.run(_run())
    for idx, ts in enumerate(stamps):
        window_count = sum(1 for t in stamps if (ts - 1.0 + 1e-6) < t <= ts + 1e-6)
        assert window_count <= 9, f"Window ending at index {idx} exceeded 9/sec"


def test_429_triggers_retry_with_backoff(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sec, "SEC_CACHE", tmp_path)
    delays: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(sec.asyncio, "sleep", _fake_sleep)

    calls = {"count": 0}

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResp:
        calls["count"] += 1
        if calls["count"] <= 2:
            return _FakeResp(429, {"error": "throttle"})
        return _FakeResp(200, {"entity": {"cik": "0001"}, "facts": {"us-gaap": {}}})

    monkeypatch.setattr(sec.requests, "get", _fake_get)

    async def _run() -> dict[str, Any]:
        limiter = sec.TokenBucketRateLimiter(rate=1e6, capacity=1e6)
        return await sec.pull_company_facts("0001", cache_hours=0, retries=3, backoff=1, limiter=limiter)

    facts = asyncio.run(_run())
    assert calls["count"] == 3
    assert delays == [1.0, 2.0]
    assert facts.get("entity", {}).get("cik") == "0001"


def test_cached_cik_not_refetched(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sec, "SEC_CACHE", tmp_path)
    cache_path = tmp_path / "companyfacts_0002.json"
    cache_path.write_text(json.dumps({"entity": {"cik": "0002"}}), encoding="utf-8")

    def _should_not_call(*args: Any, **kwargs: Any) -> _FakeResp:
        raise AssertionError("HTTP call should not be made for fresh cache")

    monkeypatch.setattr(sec.requests, "get", _should_not_call)

    async def _run() -> dict[str, Any]:
        limiter = sec.TokenBucketRateLimiter(rate=1e6, capacity=1e6)
        return await sec.pull_company_facts("0002", cache_hours=168, retries=3, backoff=1, limiter=limiter)

    facts = asyncio.run(_run())
    assert facts.get("entity", {}).get("cik") == "0002"


def test_partial_run_resumable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(sec, "SEC_CACHE", tmp_path)
    calls: dict[str, int] = defaultdict(int)

    def _fake_get(url: str, headers: dict[str, str], timeout: int) -> _FakeResp:
        cik = url.split("CIK", 1)[1].split(".json", 1)[0]
        calls[cik] += 1
        return _FakeResp(200, {"entity": {"cik": cik}, "facts": {"us-gaap": {}}})

    monkeypatch.setattr(sec.requests, "get", _fake_get)

    async def _run() -> None:
        limiter = sec.TokenBucketRateLimiter(rate=1e6, capacity=1e6)
        # First run interrupted after one completed CIK.
        await sec.pull_company_facts("0003", cache_hours=168, retries=3, backoff=1, limiter=limiter)
        # Restart run should skip completed CIK and fetch only new CIK.
        await sec.pull_company_facts("0003", cache_hours=168, retries=3, backoff=1, limiter=limiter)
        await sec.pull_company_facts("0004", cache_hours=168, retries=3, backoff=1, limiter=limiter)

    asyncio.run(_run())
    assert calls["0003"] == 1
    assert calls["0004"] == 1
