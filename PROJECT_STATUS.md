# PROJECT STATUS

> **Last Updated:** 2026-02-05 14:53 CET  
> **Updated By:** Antigravity Agent  
> **Version:** 2.1  

---

## 1. Project Overview

### Purpose
**Personal Investor Assistant** is a privacy-focused, local-first investment portfolio analytics platform. It compares portfolio performance (TWR/MWR) against benchmarks like S&P 500 and Nasdaq 100.

### Tech Stack

| Layer | Technology | Notes |
|-------|------------|-------|
| **Frontend (Web)** | Next.js 16, React 19, Tailwind 4, TypeScript | `/web` directory |
| **Frontend (Legacy)** | Streamlit ≥1.32, Plotly | `streamlit_app.py`, `/pages` |
| **Backend API** | FastAPI + Uvicorn | `src/api/server.py` (900+ lines, 58 endpoints) |
| **User Data** | SQLite | `data/user.db` (portfolios, trades, runs) |
| **Market Data** | DuckDB + Parquet | `/market_data`, `data/market_cache/` |
| **Remote Storage** | Supabase (PostgreSQL + Storage) | `/storage_supabase` |
| **Migrations** | Alembic | `/alembic` |
| **Testing** | pytest, Vitest | `/tests`, `/web/vitest.config.ts` |
| **Deployment** | Render (backend), Vercel (frontend) | `docs/DEPLOYMENT.md` |

### Architecture Overview
```
┌─────────────────┐     ┌─────────────────┐
│   Next.js Web   │────▶│   FastAPI API   │
│    (Vercel)     │     │    (Render)     │
└─────────────────┘     └────────┬────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
      ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
      │   SQLite     │   │   DuckDB/    │   │   Supabase   │
      │  (user.db)   │   │   Parquet    │   │  (remote)    │
      └──────────────┘   └──────────────┘   └──────────────┘
```

### Development Setup
```bash
# Clone and setup
git clone https://github.com/donatdermaku/personal-investor-assistant.git
cd personal-investor-assistant
make setup

# Run Streamlit UI
make run

# Run FastAPI backend
uvicorn src.api.server:app --reload

# Run Next.js frontend
cd web && npm run dev

# Run tests
make verify
```

---

## 2. Current Status

| Metric | Value |
|--------|-------|
| **Project Health** | 🟢 **Beta** - Active Development |
| **Version** | 2.0 |
| **Main Branch** | `fa46ece` |
| **Recent Activity** | 14 PRs merged in last 30 days |

---

## 3. Features & Implementation Status

### Core Analytics
| Feature | Status | Implementation | Related Files |
|---------|--------|----------------|---------------|
| TWR/MWR Performance | ✅ Complete | Time/Money-weighted returns | `src/portfolio.py`, `src/analytics/` |
| Benchmark Comparison | ✅ Complete | SPY, QQQ, custom tickers | `src/analytics/comparative.py` |
| Risk Metrics | ✅ Complete | Sharpe, drawdown, volatility | `src/analytics/risk.py`, `src/analytics/rolling.py` |
| Attribution Analysis | ✅ Complete | Sector/ticker attribution | `src/analytics/attribution.py` |
| Correlation Matrix | ✅ Complete | Assets + benchmark correlation | `src/analytics/correlation.py` |
| Macro Context | ✅ Complete | CPI, rates, regime signals | `src/analytics/macro.py` |

### Data Management
| Feature | Status | Implementation | Related Files |
|---------|--------|----------------|---------------|
| SQLite User Store | ✅ Complete | Portfolios, trades, runs | `storage/repo.py`, `storage/models.py` |
| Market Data Cache | ✅ Complete | Yahoo Finance, FRED | `market_data/yahoo.py`, `market_data/fred.py` |
| Persistent Cache | ✅ Complete | Supabase storage with index | `market_data/persistent_cache.py`, `storage/cache_index.py` |
| Data Contracts | ✅ Complete | Schema validation | `market_data/contracts.py` |
| Coverage Semantics | ✅ Complete | Data availability signals | `src/coverage.py`, `docs/COVERAGE_SEMANTICS.md` |

### UI & API
| Feature | Status | Implementation | Related Files |
|---------|--------|----------------|---------------|
| Next.js Dashboard | ✅ Complete | Overview, performance, risk, holdings | `web/src/app/(routes)/` |
| Streamlit UI | ✅ Complete | Legacy full-featured UI | `streamlit_app.py`, `pages/` |
| FastAPI Endpoints | ✅ Complete | 58 endpoints | `src/api/server.py` |
| Run Creation | ✅ Complete | Upload CSV or demo data | API `/runs` |
| Export Artifacts | ✅ Complete | JSON, CSV, HTML reports | `src/streamlit_export.py` |

### Infrastructure
| Feature | Status | Implementation | Related Files |
|---------|--------|----------------|---------------|
| Alembic Migrations | ✅ Complete | 3 migrations | `alembic/versions/` |
| Diagnostics Engine | ✅ Complete | Rule-based data quality checks | `src/diagnostics/` |
| Docker Support | ✅ Complete | Containerized deployment | `Dockerfile` |
| Render Deployment | ✅ Complete | Backend hosting | `render.yaml` |
| Vercel Deployment | ✅ Complete | Frontend hosting | `web/` |

---

## 4. Recent Changes (Last 30 Days)

### [2026-02-11] - Refactoring Phase 6 (In Progress)
**Production Readiness: Monitoring + Security Baseline**
- Agent: Codex
- Branch: `feature/refactoring-phase6`
- **Phase 6.1: Monitoring Dashboards** ✅ Implemented
  - Added `GET /ops/health` in `src/api/server.py` with uptime, memory RSS, DB status, cache stats, rate-limit config, latest run metadata
  - Added Operations page at `web/src/app/(routes)/operations/page.tsx`
  - Added route bridge at `web/app/(routes)/operations/page.tsx` and nav item in `web/src/components/nexus/Sidebar.tsx`
- **Phase 6.2: Rate Limiting and Security** ✅ Implemented (baseline)
  - Added sliding-window rate limiting middleware in `src/api/server.py` (env-driven)
  - Added security headers across API responses (nosniff, frame deny, referrer policy, permissions policy)
  - Added tests in `tests/test_api_server.py` for ops health and rate-limit enforcement
- **Phase 6.3: Performance Testing and Optimization** 🚧 Started
  - Added `scripts/perf_smoke.py` and `make perf-smoke` for repeatable latency probes
  - Added documented thresholds + optimization plan in `docs/PHASE6_3_PERFORMANCE.md`
  - Implemented timing instrumentation and lightweight ops health optimization
  - Post-deploy benchmarks executed against Cloud Run revision `00007`
  - Remaining work: configure Cloud Monitoring alerts/dashboards using documented thresholds
- UX adjustment:
  - Removed Operations page from user-visible sidebar navigation; route remains available for internal/admin usage
- Verification:
  - ✅ `pytest tests/test_api_server.py -q`
  - ✅ `npm --prefix web run lint`
  - ✅ `npm --prefix web run test`
  - ✅ `npm --prefix web run build`

### [2026-02-11] - Refactoring Phase 5
**UX Polish: Error Clarity, Progress Indicators, and UI Refinement**
- Agent: Codex
- Branch: `feature/refactoring-phase5`
- **Phase 5.1: Improve Error Messages** ✅ Complete
  - Added structured API error parsing and status-aware fallbacks in `web/src/lib/api.ts`
  - Improved route-level error descriptions in `web/src/app/(routes)/overview/page.tsx`, `web/src/app/(routes)/performance/page.tsx`, `web/src/app/(routes)/risk/page.tsx`, `web/src/app/(routes)/holdings/page.tsx`
  - Surfaced actual provider error text in `web/src/components/nexus/ContextPanel.tsx`
- **Phase 5.2: Add Progress Indicators** ✅ Complete
  - Added loading progress state (`loadingMessage`, `loadingProgress`) in `web/src/components/nexus/NexusProvider.tsx`
  - Added top-bar refresh progress UI in `web/src/components/nexus/TopBar.tsx`
  - Added run creation progress bar + step messaging in `web/src/components/nexus/RunCreationModal.tsx`
- **Phase 5.3: Polish Frontend Components** ✅ Complete
  - Refined empty states in `web/src/components/nexus/EmptyState.tsx`
  - Added subtle visual polish (background gradients + hover lift) in `web/src/app/globals.css`
  - Extended run creation typings for warning visibility in `web/src/types/nexus.ts`
- Testing performed:
  - ✅ `npm --prefix web run lint`
  - ✅ `npm --prefix web run test`
  - ✅ `npm --prefix web run build`
- Status: ✅ Complete - Ready to merge

### [2026-02-05] - Refactoring Phases 1-4
**Production-Ready Refactoring: Bug Fixes, Architecture, Caching, ETL**
- Agent: Antigravity  
- Branches: `feature/refactoring-phase1` → `feature/refactoring-phase4`
- **Phase 1: Critical Bug Fixes** ✅ Merged
  - Fixed market data "date column" bug with defensive validation in `market_data/yahoo.py` and `market_data/store.py`
  - Enhanced error handling in `src/api/server.py` with structured responses (HTTP 422 for validation errors)
  - All 117 tests passing
- **Phase 2: Architecture Cleanup** ✅ Merged
  - Removed Streamlit dependencies (streamlit, streamlit-aggrid) from requirements.txt
  - Extracted service layer: `src/services/market_data_service.py`, `src/services/portfolio_service.py`
  - Implemented repository pattern: `src/repositories/portfolio_repository.py`, `src/repositories/run_repository.py`
  - Added structured logging with structlog: `src/core/logging.py` (JSON logs for production)
  - Added mypy type checking configuration
  - Reduced server.py by 100+ lines
- **Phase 3: Database & Caching** ✅ Merged
  - Added database indexes via Alembic migration `20250204_0001_add_performance_indexes.py`
  - Implemented cache invalidation: `delete_cache_entry()`, `clear_stale_cache()`, `get_cache_age()`, `is_cache_fresh()`
  - Added auto-retry mechanism for MARKET_DATA_STALE errors in `market_data/store.py`
  - Created `CacheService` with monitoring methods: `get_cache_stats()`, `get_stale_caches()`, `clear_all_stale_caches()`
  - Admin cache endpoints: GET /admin/cache/stats, POST /admin/cache/clear-stale, GET /admin/cache/stale
- **Phase 4: ETL & Automation** ✅ Merged
  - Created `MarketDataRefreshService` for automated data refresh with `get_active_tickers()`, `refresh_market_data()`, `validate_refresh_results()`
  - Added 3 admin endpoints: POST /admin/refresh-market-data, GET /admin/refresh-status, POST /admin/backfill-market-data
  - Implemented `MarketDataValidator` with 5 data quality rules: no nulls, OHLC logic, positive volume, no duplicates, valid dates
  - Added `log_critical_error()` for Cloud Monitoring alerting with structured logging
- Testing performed:
  - ✅ All 117 tests passing across all phases
  - ✅ End-to-end testing with 1586-row CSV (30 tickers), all metrics calculated
  - ✅ Cache auto-retry mechanism verified - detects stale data, clears, and refetches successfully
  - ✅ Refresh endpoint tested with 30 tickers: 100% success rate
- Status: Phases 1-4 merged to main

### [2026-02-04] - Stability/Robustness
**Comprehensive Stability & Robustness Audit**
- Agent: Antigravity  
- Branch: `stability-audit-2026-02-04`
- Files modified:
  - `src/portfolio.py` - added division by zero protection in `align_benchmark`
  - `src/api/server.py` - added file size validation (500/2000 row limits), failed ticker reporting
- Files created:
  - `docs/INPUT_CONTRACT.md` - formal CSV format specification  
  - `tests/unit/test_edge_cases.py` - 11 new edge case tests
- Changes:
  - **Critical Fix**: Added check for zero or NaN first price in `align_benchmark` to prevent division by zero crashes
  - **File Size Validation**: Hard limit (2000 rows) and warning (500 rows) to prevent OOM crashes
  - **User Transparency**: Failed ticker warnings now included in API response (previously only logged)
  - **Documentation**: Created INPUT_CONTRACT.md with CSV requirements, action types, validation rules
  - **Edge Case Tests**: Added 11 tests covering division by zero, empty portfolios, invalid data, NaN handling
- Testing performed:
  - ✅ Edge case tests: 11/11 passed
  - ✅ Full test suite: 117 passed (up from 106), 1 skipped
  - ⚠️ Linting: 66 unused imports (non-blocking, cleanup needed)
- Known issues: Phase 20.3 memory optimizations not verified on Render production
- Next agent should: Fix linting issues, verify Phase 20.3 on Render, add frontend warnings UI
- Status: ✅ Complete - All Crash Fixes Verified


### [2026-01-27] - Infrastructure/Bugfix
**Render Production Hardening (OOM & Timeouts)**
- Agent: Claude Sonnet 4
- Files modified:
  - `render.yaml` - switched to gunicorn + 120s timeout
  - `requirements.txt` - added gunicorn
  - `src/api/server.py` - incremental gc.collect(), memory logging, /admin/clear-cache endpoint
  - `storage_supabase/storage.py` - added list_files/delete_file
  - `market_data/store.py` - fail fast on stale data
- Changes:
  - Switched from Uvicorn to Gunicorn to support timeouts >30s (fixed 502 Bad Gateway)
  - Implemented incremental garbage collection during large portfolio processing
  - Added admin endpoint to clear stale cache from Supabase & local
  - Fixed issue where stale Yahoo data (2018) caused silent failures
- Status: ✅ Deployed & Verified

### [2026-01-27] - Bugfix
**Fix Large Portfolio File Upload Crash**
- Agent: Claude Sonnet 4
- Files modified:
  - `src/api/server.py` (modified) - graceful ticker fetch failure, detailed logging
  - `market_data/rate_limiter.py` (modified) - proportional row validation
  - `market_data/store.py` (modified) - proper error handling with MarketDataError
  - `src/streamlit_data.py` (modified) - memory cleanup with gc.collect()
- Changes:
  - Added exception handling for non-MarketDataError during ticker price fetch (continues with other tickers)
  - Changed min_rows validation from fixed 1000 to proportional (70% of expected trading days, min 50)
  - Added logging at RUN_TICKERS, RUN_COMPUTE_START, RUN_COMPUTE_SUCCESS, RUN_ARTIFACTS_SAVED
  - Added explicit garbage collection after DataFrame concatenation
  - Store now converts unexpected exceptions to MarketDataError
- Testing performed:
  - ✅ **Unit tests:**: 117 passed (includes edge case tests)
- Known issues: Needs Render Starter validation with large portfolio
- Next agent should: Test with large_portfolio_trades_contract_v1_bmonthend.csv on Render
- Status: ✅ Success

### [2026-01-27] - Bugfix
**Allow IPO-era caches while still blocking undersized frames**
- Agent: Codex (GPT-5)
- Files modified:
  - `market_data/rate_limiter.py` (modified)
  - `market_data/store.py` (modified)
  - `tests/unit/test_cache_validation.py` (modified)
- Changes:
  - Start-date cache guard now only fails when the frame is undersized.
  - Raised min_rows to 1000 to block tiny caches while allowing IPO histories.
- Testing performed:
  - ✅ Manual: `scripts/repro_large_portfolio.py` completed; peak RSS ~308 MB (logs captured).
- Known issues: None.
- Next agent should: Validate Render Starter (512MB) with 30 tickers; run targeted pytest.
- Status: ✅ Success

### [2026-01-26] - Refactor/Performance
**Phase 20.3 Memory-Bounded Analytics (streaming + float32)**
- Agent: Codex (GPT-5)
- Files modified:
  - `src/pipeline.py` (modified)
  - `src/analytics/streaming.py` (new)
  - `src/analytics/attribution.py` (modified)
  - `src/analytics/correlation.py` (modified)
  - `src/analytics/risk.py` (modified)
  - `src/streamlit_data.py` (modified)
  - `src/utils_memory.py` (new)
  - `market_data/store.py` (modified)
  - `market_data/rate_limiter.py` (modified)
  - `scripts/repro_large_portfolio.py` (new)
  - `tests/unit/test_streaming_utils.py` (new)
  - `tests/unit/test_cache_validation.py` (new)
- Changes:
  - Replaced wide-matrix attribution/risk/correlation paths with streaming + online covariance (bounded memory).
  - Enforced float32 casting for market prices; returns vectors are float32.
  - Added RSS logging hooks + large-portfolio repro script.
  - Strengthened cache validation (start-date guard, duplicates).
- Testing performed:
  - ⚠️ Not run (not requested).
- Known issues: Needs Render Starter validation (512MB) for large portfolios.
- Next agent should: Run `scripts/repro_large_portfolio.py` and verify RSS/outputs; validate on Render.
- Status: ⚠️ Partial

### [2026-01-26] - Feature
**Rate Limiting and Cache Validation for Yahoo Finance**
- Agent: Claude Sonnet 4
- Files modified:
  - `market_data/rate_limiter.py` (new) - throttling, retry, validation module
  - `market_data/yahoo.py` (modified) - use throttled_fetch for all API calls
  - `market_data/store.py` (modified) - add cache validation before writing
- Changes:
  - Global rate limiter: 1 request per 1.5s, thread-safe, process-wide
  - Retry logic: respects Retry-After header, exponential backoff for 5xx/timeouts
  - Cache validation: prevents poisoned cache by validating date range, row count, columns
- Testing performed:
  - ✅ Rate limiter tests: throttling works (1.58s for 2 calls)
  - ✅ Validation tests: empty df, full df, small df all handled correctly
  - ✅ Full test suite: 94 passed, 1 skipped
- Next agent should: Add warmup as one-off job (don't block boot)
- Status: ✅ Success

### [2026-01-26] - Bugfix
**Fix MARKET_DATA_MISSING_DATES error for historical portfolios**
- Agent: Claude Sonnet 4
- Files modified:
  - `market_data/store.py`, `src/coverage.py`, `tests/unit/test_coverage_summary.py`
- Changes:
  - Added `FIXED_EARLIEST_DATE = "2010-01-01"` for max-history fetching
  - Cache now validates both start AND end dates
  - Fixed NameError in `metric_status_from_coverage`
- Status: ✅ Success

### [2026-01-26] - Investigation
**Out of Memory (OOM) / 502 Error on Large Portfolios**
- Issue: CSVs with ~30+ tickers cause OOM crash (Error 502) on Render Starter (512MB RAM).
- Root Cause: `src/pipeline.py` loads 15+ years of daily data for *all* tickers simultaneously.
- Findings:
  - Small CSVs work ✅
  - Cache warmup works ✅
  - OOM occurs during `compute_app_state` (Pandas DataFrame construction).
- Status: ⚠️ Critical bottleneck
- Next Steps: **Phase 20.3 - Memory-Bounded Analytics** (See prompt for next agent)

### [2026-01-26] - Feature (#44)
**Refined Coverage Semantics & Cache Health (Phase 20.1)**
- Agent: Antigravity
- Files modified: `market_data/calendar.py`, `src/coverage.py`, `src/api/server.py`, `web/src/lib/coverageLogic.ts`
- **Canonical Calendar**: Strict market calendar implementation (Benchmark > Union > Fallback)
- **Required Start Dates**: Per-ticker history tracking to fix false "insufficient" signals for late starters
- **Available Low Coverage**: Metrics with short history now show as "Warning" instead of being hidden
- **Cache Status**: New `GET /admin/cache-status` endpoint for health monitoring
- Testing:
  - ✅ Backend: Unit tests for calendar, coverage, and API
  - ✅ Frontend: Vitest validation for coverage logic
  - ✅ Manual: End-to-end verification script `verify_e2e.py`
- Status: ✅ Complete

### [2026-01-26] - Bugfix
**Fix Timestamp vs String comparison error in macro analytics**
- Agent: Claude Sonnet 4
- Files modified:
  - `src/analytics/macro.py` (modified)
- Changes:
  - Fixed `_align_series` function to handle timezone-aware vs timezone-naive datetime comparisons
  - Ensured both series index and target dates are converted to timezone-naive DatetimeIndex before reindexing
  - Prevented "Timestamp vs str" comparison error when processing FRED data
- Testing performed:
  - ✅ Unit tests: `tests/unit/test_macro_partial.py`, `tests/unit/test_macro_unavailable.py` passed
- Status: ✅ Success

### [2026-01-26] - Bugfix
**Fix Timestamp vs String comparison error in coverage module**
- Agent: Claude Sonnet 4
- Files modified:
  - `src/coverage.py` (modified)
- Changes:
  - Fixed type mismatch on line 158 where `prices["date"].max()` could return a `Timestamp` while benchmark date comparison expected a `date` object
  - Wrapped with `_as_date()` helper to ensure consistent Python `date` type
- Testing performed:
  - ✅ Unit tests: 4/4 coverage tests passed
  - ✅ All pytest tests: 42/42 passed
- Known issues: None (pre-existing linter warnings in test files unrelated to this fix)
- Next agent should: Test CSV upload on Render deployment to confirm production fix
- Status: ✅ Success

### [2026-01-24] - Feature (#43)
**Add persistent cache, warmup, macro partials, and correlation artifacts**
- Files: `market_data/persistent_cache.py`, `storage/cache_index.py`, `src/analytics/correlation.py`, +26 files
- Added Supabase-backed persistent cache with index tracking
- Implemented correlation matrix analytics
- Enhanced macro context with partial data handling
- Status: ✅ Complete

### [2026-01-24] - Feature (#42)
**Add diagnostics layer and export**
- Files: `src/diagnostics/engine.py`, `src/diagnostics/rules.py`, `src/diagnostics/contracts.py`
- Rule-based data quality diagnostics engine
- 13 files changed, 657 insertions
- Status: ✅ Complete

### [2026-01-24] - Feature (#41)
**Add metric-level KPI coverage gating**
- Files: `src/coverage.py`, `web/src/lib/coverageLogic.ts`, `web/vitest.config.ts`
- Frontend coverage logic with Vitest tests
- UI components now respect coverage gates
- Status: ✅ Complete

### [2026-01-24] - Feature (#40)
**Add data contracts export and unavailable semantics**
- Files: `market_data/contracts.py`, `src/api/server.py`, `src/pipeline.py`
- Contracts exported to `data_contracts.json`
- Unavailable data semantics for benchmarks/macro
- Status: ✅ Complete

### [2026-01-24] - Feature (#39)
**Add corporate actions, risk-free series, and Sharpe excess returns**
- Files: `src/risk_free.py`, `market_data/store.py`, `src/analytics/rolling.py`
- DTB3 risk-free rate integration
- Corporate actions (dividends, splits) tracking
- Status: ✅ Complete

### [2026-01-24] - Feature (#38)
**Add coverage contracts, summary export, and UI wiring**
- Files: `src/coverage.py`, `docs/COVERAGE_SEMANTICS.md`, 20 files total
- Comprehensive coverage summary structure
- UI MetricCards show coverage status
- Status: ✅ Complete

### [2026-01-23] - Feature (#37)
**Add trust-first UX context and coverage signals**
- Files: `web/src/components/nexus/ContextPanel.tsx`, `web/src/lib/coverage.ts`
- Coverage-aware UI components
- Context panels for data transparency
- Status: ✅ Complete

### [2026-01-23] - Feature (#36)
**Add insight analytics layer with attribution and risk**
- Files: `src/analytics/attribution.py`, `src/analytics/comparative.py`, `src/analytics/macro.py`
- New analytics modules: attribution, comparative, macro, risk, rolling
- 15 files changed, 1000+ insertions
- Status: ✅ Complete

---

## 5. Testing Status

### Unit Tests (`tests/unit/`)
| Test File | Description | Status |
|-----------|-------------|--------|
| `test_performance_math.py` | TWR/MWR calculations | ✅ |
| `test_risk_math.py` | Volatility, Sharpe, drawdown | ✅ |
| `test_insight_analytics.py` | Attribution, risk analytics | ✅ |
| `test_coverage_summary.py` | Coverage semantics | ✅ |
| `test_correlation_matrix.py` | Correlation analytics | ✅ |
| `test_persistent_cache.py` | Cache index operations | ✅ |
| `test_macro_partial.py` | Partial macro data handling | ✅ |
| `test_calendar_alignment.py` | Trade date alignment | ✅ |
| `test_corporate_actions.py` | Dividends/splits | ✅ |
| `test_data_contracts.py` | Contract validation | ✅ |
| `test_datamanager_fallback.py` | Storage fallback logic | ✅ |
| `test_migration_idempotent.py` | Migration safety | ✅ |
| `test_validation_harness.py` | Golden test harness | ✅ |

### Integration Tests
| Test File | Description | Status |
|-----------|-------------|--------|
| `test_golden_portfolios.py` | End-to-end portfolio computation | ✅ |
| `test_golden_portfolios_extended.py` | Extended scenarios | ✅ |
| `test_export_backend_consistency.py` | Export/API parity | ✅ |
| `test_api_server.py` | API endpoint tests | ✅ |

### Frontend Tests (`web/`)
| Framework | Coverage | Status |
|-----------|----------|--------|
| Vitest | Coverage logic tests | ✅ |

### Run Tests
```bash
make test        # Run pytest
make verify      # Full verification (compile, test, lint)
cd web && npm test  # Frontend tests
```

---

## 6. Known Issues & Bugs

| Issue | Severity | Description | Workaround |
|-------|----------|-------------|------------|
| Memory-bounded analytics validation pending | Medium | Phase 20.3 streaming refactor needs Render Starter (512MB) verification | Run `scripts/repro_large_portfolio.py` locally and validate on Render |

### Platform Stability & Data Coverage Findings (Render Free Tier)
- **Data Coverage Status**: Live runs showing ~92.2% coverage (Verified with `sample_trades_full_metrics.csv`).
  - **Issue**: Metrics show "Insufficient" because `CoveragePolicy` default threshold is 95%.
  - **Cause**: Market holidays/gaps in fresh Yahoo Finance fetches.
  - **Workaround**: Lower threshold to 90% or implement persistent cache.
- **Render Ephemeral Filesystem**:
  - Cache is wiped on every deploy/restart.
  - Triggers massive Yahoo/FRED fetching on startup/first run.
  - Leads to rate limiting (429s) and missing data (e.g., Risk-Free rate failure).
- **Recommendation**: Implement `admin/warmup` hook on deploy or use external storage (Supabase) for cache.

---

## 7. Working Components ✅

| Component | Last Tested | Notes |
|-----------|-------------|-------|
| Pipeline Engine | 2026-01-24 | Full compute cycle including all analytics |
| FastAPI Server | 2026-01-24 | All 58 endpoints functional |
| Next.js Dashboard | 2026-01-24 | Overview, Performance, Risk, Holdings pages |
| Streamlit UI | 2026-01-24 | Legacy UI fully operational |
| SQLite Storage | 2026-01-24 | Portfolios, trades, runs persisted |
| Market Data Fetch | 2026-01-24 | Yahoo Finance + FRED working |
| Persistent Cache | 2026-01-24 | Supabase storage integration |
| Alembic Migrations | 2026-01-24 | All 3 migrations applied |
| Export System | 2026-01-24 | JSON, CSV, HTML exports |
| Diagnostics | 2026-01-24 | Data quality rule engine |

---

## 8. Non-Working/Broken Components ❌

| Component | Status | Issue | Blockers |
|-----------|--------|-------|----------|
| None | - | - | - |

---

## 9. Technical Debt

| Item | Priority | Notes |
|------|----------|-------|
| Streamlit → Next.js Migration | Medium | Legacy Streamlit UI still maintained alongside Next.js |
| yfinance Rate Limiting | Low | No exponential backoff for API limits |
| Test Coverage Metrics | Low | No coverage percentage tracking configured |

---

## 10. Configuration & Environment

### Required Environment Variables
```bash
# Backend (Render)
NEXUS_DB_PATH=/var/data/user.db
NEXUS_EXPORT_DIR=/var/data/exports
NEXUS_ALLOWED_ORIGINS=https://<vercel-app>.vercel.app
STORAGE_MODE=hybrid

# Frontend (Vercel)
NEXT_PUBLIC_API_URL=https://<render-service>.onrender.com

# Supabase (optional)
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_ANON_KEY=<key>
SUPABASE_SERVICE_KEY=<service-key>
```

### Database Migrations
```bash
# Apply migrations
alembic upgrade head

# Migration status
alembic current
```

| Migration | Description | Status |
|-----------|-------------|--------|
| `20250124_0001_init_supabase.py` | Supabase tables | ✅ Applied |
| `20250124_0002_add_holdings_snapshots.py` | Holdings snapshots | ✅ Applied |
| `20250124_0003_add_data_cache_index.py` | Cache index table | ✅ Applied |

---

## 11. Dependencies Status

### Python (`requirements.txt`)
| Package | Version | Notes |
|---------|---------|-------|
| pandas | Latest | Core data handling |
| numpy | <2 | Pinned for compatibility |
| yfinance | Latest | Market data source |
| duckdb | Latest | Analytical queries |
| fastapi | Latest | REST API |
| streamlit | ≥1.32 | Legacy UI |
| sqlalchemy | Latest | ORM |
| alembic | Latest | Migrations |
| psycopg2-binary | Latest | Supabase PostgreSQL |

### Node.js (`web/package.json`)
| Package | Version | Notes |
|---------|---------|-------|
| next | 16.1.4 | Web framework |
| react | 19.2.3 | UI library |
| recharts | ^3.7.0 | Chart library |
| tailwindcss | ^4 | Styling |
| vitest | ^2.1.0 | Testing |
| typescript | ^5 | Type safety |

---

## 12. Performance Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Backend cold start | ~5s | Render free tier |
| Pipeline compute | ~10-30s | Depends on portfolio size |
| API `/latest-run` | <100ms | Cached manifest |
| Market data fetch | 1-3s/ticker | Yahoo Finance |

---

## 13. Next Steps & Priorities

### Immediate (This Week)
- [ ] Complete Phase 6.3 performance benchmark run on deployed environment
- [ ] Tune thresholds and optimize top latency bottleneck identified by `scripts/perf_smoke.py`
- [ ] Validate Phase 20.3 memory-bounded pipeline on Render Starter (512MB)
- [ ] Run `scripts/repro_large_portfolio.py` and capture peak RSS
- [ ] Monitor persistent cache warmup efficiency
- [ ] Validate correlation matrix accuracy in production

### Short-term (This Month)
- [ ] Add test coverage reporting with pytest-cov
- [ ] Enhance diagnostics with more rules
- [ ] Performance optimization for large portfolios

### Long-term Roadmap
- [ ] Complete Streamlit → Next.js migration
- [ ] Multi-user support with authentication
- [ ] Real-time market data streaming
- [ ] Mobile-responsive Next.js dashboard

---

## 14. Development Notes

### Key Architectural Decisions
1. **Hybrid Storage**: SQLite for ACID transactions (user data), DuckDB/Parquet for analytics performance
2. **Coverage Semantics**: Explicit data availability signals propagated to UI
3. **Trust-First UX**: Every metric shows its data source and freshness
4. **Immutable Runs**: Each portfolio computation creates a new immutable run with all artifacts

### Code Patterns
- **Repository Pattern**: `storage/repo.py` abstracts all data access
- **Data Contracts**: `market_data/contracts.py` enforces schemas
- **Pipeline Pattern**: `src/pipeline.py` orchestrates computation flow

### Gotchas
- ⚠️ `numpy<2` constraint required for extension compatibility
- ⚠️ CORS must include exact Vercel URL with `https://`
- ⚠️ Market data cache uses business day logic (weekends skipped)

---

## Documentation Links

| Document | Path | Description |
|----------|------|-------------|
| README | `/README.md` | Quick start guide |
| Architecture | `/docs/ARCHITECTURE.md` | System design |
| Deployment | `/docs/DEPLOYMENT.md` | Render + Vercel setup |
| Market Data | `/docs/MARKET_DATA.md` | Data contracts and sources |
| Coverage Semantics | `/docs/COVERAGE_SEMANTICS.md` | Data availability rules |
| Metrics Definitions | `/docs/METRICS_DEFINITIONS.md` | KPI specifications |
| Schema | `/docs/SCHEMA.md` | Database schema |

---

## 15. Phase -1 Baseline Capture

### ENVIRONMENT
- Python version: `3.12.11`
- pandas: `2.2.3`
- numpy: `1.26.4`
- duckdb: `1.4.3`
- scipy: `1.13.1`
- sqlalchemy: `2.0.41`
- Full pip freeze: `docs/planning/pip_freeze_phase_minus_1.txt`

### BASELINE TEST RESULTS
- Command: `pytest --tb=short -q`
- Exact result: `149 passed, 1 skipped, 70 warnings in 20.20s`

### FIXTURE HASHES
- Random seed used for fixture generation: `np.random.seed(42)`
- `tests/fixtures/baseline_ledger.csv`: `817f4c68cc027784d564ee349f42f438ece64360d504370bb1b92dfa3141c18f`
- `tests/fixtures/baseline_prices.parquet`: `9e6b396c74e597549f63afe2c1ef341b3aaf2ec105435fe969b673aec3b6e3dc`

### GOLDEN METRICS SNAPSHOT (PHASE -1)
- File: `tests/fixtures/golden_metrics_phase_minus_1.json`
- Payload:
```json
{
  "twr": 0.0493976052144709,
  "mwr": -0.07242406503979686,
  "sharpe_rolling_last": 0.010716854141029198,
  "sharpe_api": 0.6222303689200788,
  "max_drawdown": -0.09112418483507145,
  "factor_tilt_value": 12.5,
  "factor_tilt_quality": 5.0,
  "factor_tilt_momentum": 5.0,
  "score_coverage_pct": null,
  "rf_coverage_pct": null
}
```

### FUNCTION INVENTORY
file: `src/analytics/rolling.py`
  - `compute_rolling_metrics()` — lines 7-46 — computes rolling volatility, rolling Sharpe, rolling drawdown — BUG: rolling Sharpe divides daily mean by annualized volatility.

file: `src/portfolio.py`
  - `compute_twr()` — lines 117-131 — daily geometric linking with external cashflows — BUG: `inf/-inf` normalized to `0.0`, which can corrupt linked return logic.
  - `compute_irr()` — lines 134-135 — thin wrapper over `_xirr` — BUG: no valuation end date parameter.
  - `compute_portfolio_from_ledger()` — lines 232-366 — builds cashflow series and valuations — BUG: MWR flow sign semantics and inclusion of internal flows (`DIVIDEND`, `INTEREST`, `FEE`).
  - `_xirr()` — lines 415-468 — Newton + finite-difference derivative + bisection fallback — BUG: hardcoded bracket and terminal value date handling.

file: `src/api/server.py`
  - `_compute_risk_metrics()` — lines 710-746 — point-in-time VaR/CVaR/volatility/Sharpe from exported returns — dependency for Phase 0 Sharpe parity check and Phase 1 RF coverage behavior.

file: `src/compute/factors.py`
  - `_build_ttm_rollup()` — lines 45-75 — pandas-based trailing four-quarter rollups for fundamentals.
  - `_calc_fundamental_metrics()` — lines 125-191 — per-ticker grouped transformations and rolling stats.
  - `compute()` — lines 203-417 — factor pipeline composition (value/quality/momentum/composite).
  - `main()` — lines 420-538 — loads full DuckDB tables and writes score outputs.

file: `src/ingest/fundamentals_sec.py`
  - `pull_company_facts()` — lines 60-77 — SEC fetch + cache with retries.
  - `main()` — lines 119-188 — sequential ticker ingestion and persistence.

file: `src/ingest/prices.py`
  - `_split_multiindex()` — lines 77-91 — splits yfinance multi-index frame per ticker.
  - `main()` — lines 94-160 — batched yfinance download and persistence.

file: `manage.py`
  - `cmd_update()` — lines 41-46 — currently aliases compute path; does not execute ingest stages.

### OPEN ISSUES
- Plan/file mismatch detected before Phase 0 start:
  - Risk-free implementation file in repository is `src/risk_free.py` (not `src/analytics/risk_free.py`).
  - Factor-tilt runtime implementation is in `src/analytics/factors.py`; plan references `src/compute/factors.py` for Phase 1.3 behavior change.
- Resolution to apply after Phase -1 gate: treat these as execution-path ambiguities and log final handling under `PLAN DEVIATIONS` once Phase 0 begins.


### Phase 0.1 Status
- Verified `rolling_vol = std(ddof=1) * sqrt(252)` in `src/analytics/rolling.py`.
- Updated rolling Sharpe to annualized form: `(rolling_mean * 252) / rolling_vol`.
- Added targeted tests: `tests/test_rolling.py`.
- Gate result: `2 passed` (`pytest --tb=short -q tests/test_rolling.py`).

### Phase 0.2 Status
- Updated MWR external-flow semantics in `src/portfolio.py`:
  - External MWR flows include only `DEPOSIT` and `WITHDRAWAL`.
  - `DEPOSIT -> negative`, `WITHDRAWAL -> positive`.
  - `compute_irr(..., valuation_end_date=...)` now requires valuation end date and appends terminal value at that date.
- Added targeted tests: `tests/test_mwr.py`.
- Gate result: `4 passed` (`pytest --tb=short -q tests/test_mwr.py`).

### TWR SUB-PERIOD CONVENTION
Worked example (required convention before implementation):
- Phase 1: portfolio falls from `100` to `0` with no external inflow during loss interval.
  - `TWR_1 = (0 / 100) - 1 = -1.0`.
- Phase 2: new capital injected (`+100`) and portfolio grows from `100` to `200`.
  - `TWR_2 = (200 / 100) - 1 = +1.0`.
- Linked total:
  - `TWR_total = (1 + TWR_1) * (1 + TWR_2) - 1`
  - `= (1 - 1.0) * (1 + 1.0) - 1`
  - `= 0 * 2 - 1 = -1.0`.

Therefore a complete wipeout in an earlier linked sub-period is permanent in the linked TWR chain.

### Phase 0.3 Status
- Implemented zero-balance sub-period relinking in `compute_twr()`.
- Undefined periods (`V_(t-1) == 0` and no cashflow) now stay `NaN` in raw daily TWR output and are excluded from geometric linking.
- Added targeted tests: `tests/test_twr.py`.
- Gate result: `4 passed` (`pytest --tb=short -q tests/test_twr.py`).

### Phase 0 Completion Gate
- Full suite command: `pytest --tb=short -q`
- Result: `159 passed, 1 skipped, 70 warnings in 5.77s`
- Baseline floor preserved: no previously passing test remains failing.

### Phase 0 Golden Snapshot
- Created: `tests/fixtures/golden_metrics_phase_0.json`
- Delta vs `tests/fixtures/golden_metrics_phase_minus_1.json`:
  - `twr`: `0.0493976052144709 -> 0.0493976052144709` (no change; expected for this fixture).
  - `mwr`: `-0.07242406503979686 -> 0.04973609575086422` (intentional fix from corrected external-flow filtering/sign convention + terminal-date timing).
  - `sharpe_rolling_last`: `0.010716854141029198 -> 2.700647243539358` (intentional annualization fix in rolling Sharpe).


### PLAN DEVIATIONS
- Original instruction: Phase 1.2 target file `src/analytics/risk_free.py`.
  - Blocker discovered: repository risk-free implementation is located at `src/risk_free.py`; `src/analytics/risk_free.py` does not exist.
  - Resolution chosen: implemented the Phase 1.2 changes in `src/risk_free.py` and propagated to `src/analytics/rolling.py` and `src/api/server.py` as specified.
- Original instruction: Phase 1.3 target file `src/compute/factors.py` for factor tilt bias correction.
  - Blocker discovered: live portfolio tilt computation is implemented in `src/analytics/factors.py` and called from `src/pipeline.py`.
  - Resolution chosen: implement the Phase 1.3 tilt-bias fix in `src/analytics/factors.py` (runtime path), keeping `src/compute/factors.py` unchanged for the scoring engine.


### EXCEPTIONS
- File touched outside strict per-subphase target list for dependency wiring: `src/streamlit_export.py`.
  - Reason: surface new non-fatal `PortfolioResult.warnings` in exported summary payload so IRR `no_root/ambiguous_multi_root` statuses are user-visible.
  - Scope: added `warnings` field to summary JSON payload only; no analytics math changes.

### Phase 1.1 Status
- Replaced custom Newton/Bisection IRR solver with interval scan + `scipy.optimize.brentq` in `src/portfolio.py`.
- Added typed IRR result object:
  - `IRRResult(value, status, message)` where status is `ok | no_root | ambiguous_multi_root`.
- Caller wiring:
  - `compute_portfolio_from_ledger()` now handles IRR statuses explicitly and stores non-fatal IRR status in `PortfolioResult.warnings`.
- Added targeted tests: `tests/test_irr_solver.py`.
- Gate result: `4 passed` (`pytest --tb=short -q tests/test_irr_solver.py`).

### Phase 1.2 Status
- Removed silent risk-free zero-fill in `src/risk_free.py`.
- Updated rolling Sharpe in `src/analytics/rolling.py` to compute excess returns only where RF is present.
- Updated API risk metrics in `src/api/server.py`:
  - Sharpe/volatility from excess returns only on RF-overlap.
  - Added `rf_coverage_pct` in payload.
- Added targeted tests: `tests/test_risk_free.py`.
- Gate result: `4 passed` (`pytest --tb=short -q tests/test_risk_free.py`).

### Phase 1.3 Status
- Implemented factor-tilt bias correction in runtime tilt path `src/analytics/factors.py`:
  - No post-drop renormalization of covered holdings.
  - Added `score_coverage_pct` and `score_coverage_by_factor` to summary.
  - Logs warning when coverage `< 0.5`.
- Added/updated targeted tests in `tests/unit/test_factors.py`.
- Gate result: `5 passed` (`pytest --tb=short -q tests/unit/test_factors.py`).

### Phase 1 Completion Gate
- Full suite command: `pytest --tb=short -q`
- Result: `170 passed, 1 skipped, 70 warnings in 7.71s`
- Created: `tests/fixtures/golden_metrics_phase_1.json`

### Phase 1 Golden Comparison
Comparison against `tests/fixtures/golden_metrics_phase_0.json` on baseline fixture:
- `twr`: unchanged (`0.0493976052144709`).
- `mwr`: `0.04973609575086422 -> 0.049736095726908874` (delta `~2.40e-11`, within tolerance).
- `sharpe_rolling_last`: unchanged (`2.700647243539358`).
- `sharpe_api`: unchanged (`0.6222303689200788`).
- New metadata field from Phase 1.3: `score_coverage_pct = 1.0`.

### Phase 2.1 Status
- Implemented DuckDB aggregation push-down in `src/compute/factors.py`:
  - Removed pandas `_build_ttm_rollup()` and replaced TTM rollups with window SQL in `_calc_fundamental_metrics()`.
  - Pushed 252-day momentum computation into DuckDB (`LAG(..., 252)`) in `_calc_price_metrics()`.
  - In `main()`, replaced full-table `SELECT *` with universe-filtered projection:
    - `prices_daily`: `ticker, date, adj_close`
    - `fundamentals_quarterly`: factor-required projected columns only.
- Numerical identity verification:
  - Added `tests/test_factors_pushdown.py`.
  - TTM comparison vs legacy pandas rollup: max abs diff `4.547473508864641e-13` (within raw tolerance `1e-12`).
  - 252-day momentum comparison vs manual baseline: exact within `1e-12`.
- Memory benchmark (tracemalloc, synthetic 600 tickers x 16 quarters):
  - Legacy peak: `26,364,730` bytes.
  - Phase 2.1 peak: `18,976,434` bytes.
  - Reduction: `28.02%`.

### Phase 2.2 Status
- Refactored `src/ingest/fundamentals_sec.py`:
  - Added async ingestion path with `asyncio.gather(...)`.
  - Added `TokenBucketRateLimiter` and `await limiter.acquire()` before each SEC call.
  - Added retry behavior for `429/503` with exponential backoff (`1s`, `2s`, `4s` with configured base and max retries).
  - Added cache freshness skip semantics with optional CLI flag `--force-refresh`.
  - Kept cache default at 7 days (`sec_cache_hours` default `168`).
- Added `tests/test_sec_ingestion.py`:
  - `test_rate_limiter_enforces_limit`
  - `test_429_triggers_retry_with_backoff`
  - `test_cached_cik_not_refetched`
  - `test_partial_run_resumable`
- Subphase gate result:
  - `pytest --tb=short -q tests/test_sec_ingestion.py` (included in targeted Phase 2 run) passed.

### Phase 2.3 Status
- Updated `src/ingest/prices.py::_split_multiindex()`:
  - Replaced broad per-ticker membership checks with vendor presence pre-filtering and `xs(..., drop_level=True)` extraction.
- Added `tests/test_prices_split.py`:
  - Verifies numerical identity against legacy splitter on 10-ticker MultiIndex batch.
  - Max numeric diff observed: `0.0` (within `1e-12` tolerance).
- Memory benchmark (tracemalloc):
  - Workload: 500 requested tickers, 10 returned tickers, 1,200 business days.
  - Legacy peak: `993,740` bytes.
  - Phase 2.3 peak: `967,831` bytes.
  - Reduction: `2.61%`.

### PLAN DEVIATIONS
- Original instruction: Phase 2.2 suggested limiter config `rate=9.0`, `capacity=9.0`.
  - Blocker discovered: burst capacity of `9.0` can exceed strict per-second cap under sliding-window validation.
  - Resolution chosen: keep `rate=9.0` and set runtime limiter capacity to `1.0` for strict `<=9 req/sec` compliance in tests and ingestion runtime.

### Phase 2 Completion Gate
- Full suite command: `pytest --tb=short -q`
- Result: `177 passed, 1 skipped, 71 warnings in 13.23s`
- Created: `tests/fixtures/golden_metrics_phase_2.json`
- Golden comparison (`phase_2` vs `phase_1`):
  - `twr`: unchanged (`0.0493976052144709`)
  - `mwr`: unchanged (`0.049736095726908874`)
  - `sharpe_rolling_last`: unchanged (`2.700647243539358`)
  - `sharpe_api`: unchanged (`0.6222303689200788`)
  - `max_drawdown`: unchanged (`-0.09112418483507145`)
  - `factor_tilt_value`: unchanged (`12.5`)
  - `factor_tilt_quality`: unchanged (`5.0`)
  - `factor_tilt_momentum`: unchanged (`5.0`)
  - `score_coverage_pct`: unchanged (`1.0`)
  - `rf_coverage_pct`: unchanged (`null`)

### Phase 3.1 Status
- Updated `manage.py::cmd_update(args)` to run staged workflow in strict order:
  - `ingest_universe -> ingest_prices -> ingest_fundamentals -> compute_factors -> compute_analytics`.
- Added explicit failure halting semantics:
  - Any stage exception exits non-zero and prevents downstream stages from executing.
- Added CLI flags to `update`:
  - `--compute-only` (skips ingest/factor stages and runs compute analytics only).
  - `--dry-run` (prints planned stage names without execution).
- Added tests: `tests/test_cli.py`:
  - `test_update_calls_ingest_before_compute`
  - `test_ingest_failure_halts_pipeline`
  - `test_compute_only_skips_ingest`
  - `test_dry_run_no_execution`
- Subphase gate result:
  - `pytest --tb=short -q tests/test_cli.py` -> `4 passed`.

### Phase 3 Completion Gate
- Full suite command: `pytest --tb=short -q`
- Result: `181 passed, 1 skipped, 71 warnings in 13.18s`
- Created: `tests/fixtures/golden_metrics_phase_3.json`
- Exact identity check:
  - `golden_metrics_phase_3.json` matches `golden_metrics_phase_2.json` byte-for-byte (`cmp` exit `0`).

### Final Verification
- End-to-end run executed:
  - `python manage.py update --compute-only`
  - Result: succeeded (`exit 0`) and produced artifacts for run `6eb4cbb5-9c19-4782-98a4-d4c17eb04359`.
  - Note: network-restricted environment caused expected Yahoo DNS warnings during benchmark/market fetch; pipeline still completed.
- Fixture hash check (must remain unchanged):
  - `tests/fixtures/baseline_ledger.csv`: `817f4c68cc027784d564ee349f42f438ece64360d504370bb1b92dfa3141c18f` (unchanged)
  - `tests/fixtures/baseline_prices.parquet`: `9e6b396c74e597549f63afe2c1ef341b3aaf2ec105435fe969b673aec3b6e3dc` (unchanged)
- Metric deltas vs `golden_metrics_phase_minus_1.json` (bug-fix rationale):
  - `mwr` changed from `-0.07242406503979686` to `0.049736095726908874` because Phase 0 corrected IRR cashflow semantics (external flow filtering/sign convention) and terminal valuation date handling.
  - `sharpe_rolling_last` changed from `0.010716854141029198` to `2.700647243539358` because Phase 0 fixed rolling Sharpe annualization to align with annualized volatility.
  - `score_coverage_pct` changed from `null` to `1.0` because Phase 1 introduced explicit factor-score coverage reporting instead of implicit omission.

### REFACTOR COMPLETE
- Phases completed:
  - `Phase -1`, `Phase 0`, `Phase 1`, `Phase 2`, `Phase 3`, and final verification.
- Total tests added:
  - 10 new test files during phased refactor (`tests/test_rolling.py`, `tests/test_mwr.py`, `tests/test_twr.py`, `tests/test_irr_solver.py`, `tests/test_risk_free.py`, `tests/test_factors_pushdown.py`, `tests/test_sec_ingestion.py`, `tests/test_prices_split.py`, `tests/test_cli.py`, plus additional phase-specific updates to existing suites).
- Open issues remaining:
  - None blocking; known warnings are deprecation/network-environment related and not correctness regressions.
- Financial metrics changed and why:
  - `MWR` and rolling `Sharpe` changed due to intentional mathematical bug fixes in Phase 0.
  - Coverage metadata (`score_coverage_pct`) became explicit in Phase 1.
  - Other locked metrics remained stable through Phases 2–3, confirming architecture/UX changes were non-regressive.

### REMEDIATION NOTES
- Gap identified post-review:
  - `src/analytics/rolling.py::compute_rolling_metrics()` did not emit `rf_coverage_pct` in rolling output.
- Fix applied:
  - Added `rf_coverage_pct` scalar computation from aligned `rf_daily` non-null fraction.
  - Broadcast `rf_coverage_pct` into the returned rolling output DataFrame as a column.
  - Kept return type unchanged (`pd.DataFrame`).
- Additional compatibility handling:
  - `compute_rolling_metrics()` now also accepts `pd.Series` inputs for `performance` and `risk_free_series` to support the required remediation tests while preserving existing DataFrame behavior.
  - Missing `drawdown` now yields `NaN` rolling drawdown instead of raising.
- Required tests added:
  - `test_rf_coverage_pct_present_in_rolling_output`
  - `test_rf_coverage_pct_none_when_no_rf`
- Targeted test gate:
  - `pytest tests/test_risk_free.py -v` -> `6 passed`.
- Full suite gate:
  - `pytest --tb=short -q` -> `183 passed, 1 skipped`.
  - Note: baseline previously reported `181 passed`; count increased by exactly 2 due the two newly required remediation tests. No regressions observed.
- Baseline fixture hashes re-verified unchanged:
  - `tests/fixtures/baseline_ledger.csv`: `817f4c68cc027784d564ee349f42f438ece64360d504370bb1b92dfa3141c18f`
  - `tests/fixtures/baseline_prices.parquet`: `9e6b396c74e597549f63afe2c1ef341b3aaf2ec105435fe969b673aec3b6e3dc`
- Golden metrics regeneration scope:
  - Updated only: `tests/fixtures/golden_metrics_phase_1.json`, `tests/fixtures/golden_metrics_phase_2.json`, `tests/fixtures/golden_metrics_phase_3.json`.
  - Unchanged by policy: `tests/fixtures/golden_metrics_phase_minus_1.json`, `tests/fixtures/golden_metrics_phase_0.json`.
- Golden metrics consistency check:
  - For phases 1–3, only `rf_coverage_pct` changed (`null -> 1.0`).
  - All other fields remained exactly unchanged.
- Updated golden file hashes:
  - `tests/fixtures/golden_metrics_phase_1.json`: `2090018659a78a3c50363eca4e772c4ff64dc65d1ecb6bd7f71a5c4936f2985b`
  - `tests/fixtures/golden_metrics_phase_2.json`: `2090018659a78a3c50363eca4e772c4ff64dc65d1ecb6bd7f71a5c4936f2985b`
  - `tests/fixtures/golden_metrics_phase_3.json`: `2090018659a78a3c50363eca4e772c4ff64dc65d1ecb6bd7f71a5c4936f2985b`

### REMEDIATION
- Files modified:
  - `src/analytics/rolling.py`
  - `tests/test_risk_free.py`
  - `tests/fixtures/golden_metrics_phase_1.json`
  - `tests/fixtures/golden_metrics_phase_2.json`
  - `tests/fixtures/golden_metrics_phase_3.json`
  - `PROJECT_STATUS.md`
- Remediation outcome:
  - Rolling output now includes `rf_coverage_pct` per implementation-plan requirement.
  - `rf_coverage_pct` values in golden metrics:
    - Phase 1: `1.0`
    - Phase 2: `1.0`
    - Phase 3: `1.0`

### REFACTOR COMPLETE (POST-REVIEW UPDATE)
- Post-review remediation applied for rolling RF coverage emission parity.
- No regressions introduced; remediation is scoped and validated by targeted + full test gates.
