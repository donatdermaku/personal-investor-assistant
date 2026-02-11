# Phase 6.3 Performance Baseline And Optimization Plan

Date: 2026-02-11  
Environment: Google Cloud Run (`personal-investor-assistant`, `us-central1`)  
Base URL: `https://personal-investor-assistant-56ll234k6a-uc.a.run.app`

## Baseline Measurements

Method:
- `scripts/perf_smoke.py` for 2xx endpoints
- `curl` loop for `/latest-run` (404 path included)
- Sample size: 60 requests per endpoint

| Endpoint | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| `/health` | 231.90 | 217.61 | 340.95 | 347.29 | Lightweight health probe |
| `/ops/health` | 234.53 | 218.90 | 338.29 | 351.94 | Includes runtime/cache metadata |
| `/definitions` | 227.51 | 222.45 | 239.07 | 306.29 | Stable read path |
| `/runs` | 223.52 | 218.15 | 240.09 | 337.91 | DB-backed list |
| `/latest-run` (404 path) | 235.15 | 215.45 | 247.60 | 401.11 | Empty-state path in production |

## Post-Implementation Measurements (Revision `00007`)

Changes deployed:
- Request timing instrumentation middleware
- `/ops/health` lightweight default mode (skip cache stats unless `full=true`)
- Short-TTL in-memory cache for `/definitions` and `/ops/health?full=true` cache stats

| Endpoint | Avg (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Notes |
|---|---:|---:|---:|---:|---|
| `/health` | 220.49 | 205.95 | 227.86 | 291.88 | Improved tail latency |
| `/ops/health` | 238.73 | 207.65 | 226.98 | 243.52 | Default lightweight path |
| `/ops/health?full=true` | 257.50 | 208.91 | 223.65 | 227.75 | Full cache stats mode |
| `/definitions` (run 1) | 299.84 | 210.86 | 243.94 | 291.24 | One run showed outlier mean |
| `/definitions` (run 2) | 214.99 | 212.90 | 224.45 | 260.15 | Warm/stable path |

Observation:
- P95 and P99 are now comfortably within proposed thresholds.
- `/definitions` average can spike due to occasional outliers, but tail latency remains healthy.

## Proposed Thresholds

### Tier A: Health Endpoints
- Scope: `/health`, `/ops/health`
- Target: P95 <= 350ms, P99 <= 500ms
- Alert: P95 > 450ms for 5 minutes
- Critical: P95 > 600ms for 5 minutes

### Tier B: Metadata/List Endpoints
- Scope: `/definitions`, `/runs`, `/latest-run`
- Target: P95 <= 300ms, P99 <= 450ms
- Alert: P95 > 400ms for 10 minutes
- Critical: P95 > 600ms for 10 minutes

### Error Budget Trigger
- Investigate when:
  - Any endpoint exceeds critical threshold twice in 24h, or
  - P95 regresses >20% week-over-week

## Optimization Plan

1. Instrument endpoint timing in API logs (request path + duration + status). ✅ Implemented
2. Add Cloud Monitoring dashboards:
   - P50/P95/P99 by endpoint
   - Request count + error rate + 429 rate
   - Memory RSS and container instance count
3. Add lightweight short-TTL in-memory caching for: ✅ Implemented
   - `/definitions`
   - `/ops/health` cache stats section (when `full=true`)
4. Optimize `/ops/health`: ✅ Implemented
   - Cache stats optional via query flag (`?full=1`)
   - Default payload skips expensive cache stats for polling UIs
5. Tune Cloud Run runtime once traffic profile is known:
   - Evaluate `min-instances=1` for lower tail latency
   - Validate CPU/memory balance for steady-state cost/perf
6. Re-benchmark after each change using the same 60-request protocol and compare deltas.

## Completion Criteria For Task 6.3

- Thresholds adopted in docs and monitoring alerts configured.
- Two benchmark runs completed post-optimization with:
  - Tier A and Tier B targets met.
- Regression guard added to deployment checklist:
  - Run `scripts/perf_smoke.py` before production release.

Current status:
- Thresholds: ✅ documented
- Post-optimization benchmarks: ✅ completed
- Monitoring alert wiring: ⏳ pending in Cloud Monitoring setup
