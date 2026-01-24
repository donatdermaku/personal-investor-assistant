# Data Availability & Cache Semantics

Nexus uses a persistent cache for macro and benchmark data to avoid empty panels after backend restarts.
Cached frames are stored in Supabase Storage with an index in Postgres.

## Cache Status

Each cached dataset has a status:

- `fresh`: cache is within TTL and safe to use.
- `stale`: last-known-good data is served due to refresh failure or TTL expiry.
- `error`: no usable cache available.

The cache index lives in `data_cache_index` with metadata:
source, key, asof_date, updated_at, ttl_seconds, status, coverage_pct, storage_path, error_code, error_message.

## Last-Known-Good Behavior

If a refresh fails, Nexus serves the prior cached data and marks the cache as `stale`.
Errors are recorded in the cache index and surfaced in macro warnings.

## Partial vs Unavailable

- `partial`: enough data to compute some tags or metrics, but not all.
- `unavailable`: insufficient data to compute any meaningful output.

Artifacts always exist and include explicit `status` and `reasons` fields.

## Warmup

The `/admin/warmup` endpoint refreshes macro and benchmark caches.
Use header `X-ADMIN-KEY` with `ADMIN_WARMUP_KEY` env var.

Payload:

```json
{
  "benchmarks": ["SPY"],
  "force": false
}
```

Warmup writes a report to `system/warmup/<timestamp>.json` in storage.
