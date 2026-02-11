#!/usr/bin/env python
"""
Lightweight API performance smoke test.

Example:
  python scripts/perf_smoke.py --base-url http://localhost:8000 --path /health --requests 30
"""

from __future__ import annotations

import argparse
import statistics
import time
import urllib.error
import urllib.request


def run_probe(url: str, request_count: int, timeout: float) -> list[float]:
    timings_ms: list[float] = []
    for _ in range(request_count):
        started = time.perf_counter()
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            # Force full read for realistic timing.
            _ = resp.read()
        elapsed_ms = (time.perf_counter() - started) * 1000
        timings_ms.append(elapsed_ms)
    return timings_ms


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(round((p / 100) * (len(ordered) - 1)))
    return ordered[idx]


def main() -> int:
    parser = argparse.ArgumentParser(description="Nexus API performance smoke test")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--path", default="/health", help="Endpoint path to probe")
    parser.add_argument("--requests", type=int, default=25, help="Number of requests")
    parser.add_argument("--timeout", type=float, default=10.0, help="Request timeout (seconds)")
    args = parser.parse_args()

    url = f"{args.base_url.rstrip('/')}{args.path}"
    try:
        timings = run_probe(url, args.requests, args.timeout)
    except urllib.error.URLError as exc:
        print(f"Request failed for {url}: {exc}")
        return 1

    p50 = percentile(timings, 50)
    p95 = percentile(timings, 95)
    p99 = percentile(timings, 99)
    avg = statistics.mean(timings)
    print(f"Endpoint: {url}")
    print(f"Requests: {len(timings)}")
    print(f"Average: {avg:.2f}ms")
    print(f"P50: {p50:.2f}ms")
    print(f"P95: {p95:.2f}ms")
    print(f"P99: {p99:.2f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
