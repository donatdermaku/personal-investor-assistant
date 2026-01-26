# PROJECT STATUS

> **Last Updated:** 2026-01-26 18:14 CET  
> **Updated By:** Claude Sonnet 4  
> **Version:** 2.0  

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
| None reported | - | - | - |

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
