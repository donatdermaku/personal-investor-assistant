# Project Nexus: Revised Implementation Plan

> **Goal:** Validated "Zero-Cost" Beta Launch (50 users)  
> **Constraint:** $0 incremental spend. Supabase free tier + Cloud Run free tier.  
> **Based on:** Code-level gap analysis (2026-02-12)

---

## Architecture Decision: Supabase-First

| Data Type | Storage | Rationale |
|-----------|---------|-----------|
| Users, Auth | Supabase Auth | Free JWT-based auth, OAuth support |
| Portfolios, Trades, Runs, Artifacts | Supabase Postgres + Storage | Multi-tenant, persistent, RLS-ready |
| Market data (prices, FRED) | Local cache + Supabase Storage backup | Shared across users, no per-user isolation needed |
| SQLite | Local dev only (`STORAGE_MODE=local`) | Remove `hybrid` mode for simplicity |

---

## Phase 0: Foundation Cleanup *(~1-2 days)*

> **Status: ~90% done.** Golden tests, yfinance isolation, and validation all exist.

### What's left:

- [ ] **0.1 — Fix `cloudbuild.yaml`**: Remove duplicate `options` block
- [ ] **0.2 — Clean repo root**: Move or `.gitignore` loose CSV files (`trades_part*.csv`, `large_portfolio_*.csv`)
- [ ] **0.3 — Simplify `STORAGE_MODE`**: Reduce from 3 modes (`db`/`files`/`hybrid`) to 2 (`local`/`supabase`). Update [datamanager.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/storage/datamanager.py)
- [ ] **0.4 — GitHub Actions CI**: Add workflow that runs `pytest` + `npm run build` + `npm run lint` on every push/PR

### Already done (skip):
- ✅ Golden test harness (7 scenarios)
- ✅ yfinance isolation + rate limiting
- ✅ CSV input validation (`validate_ledger()`)

---

## Phase 1: Value Engine — "Killer Insights" *(~3-5 days)*

> **Status: ~50% done.** Core analytics refactor is complete. Missing the differentiating insights.

### What's left:

- [ ] **1.1 — Metrics Registry enforcement**: Create [src/analytics/metrics_registry.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/analytics/metrics_registry.py) that gates unregistered metrics from API/report exposure. Wire into existing `DEFINITIONS_REGISTRY` in [definitions.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/definitions.py)
- [ ] **1.2 — HHI Concentration Index**: Pure function in `src/analytics/concentration.py`
  - `compute_hhi(weights: pd.Series) -> float`
  - Threshold tiers: diversified (<0.15), moderate (0.15-0.25), concentrated (>0.25)
  - Register in metrics registry
- [ ] **1.3 — VaR Budget comparison**: Add `compare_var_budget(portfolio_returns, benchmark_returns, alpha=0.05)` to [risk.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/analytics/risk.py). Currently VaR is computed inside `compute_risk_contributions()` but never compared to benchmark
- [ ] **1.4 — Factor Tilt computation**: Implement what's already defined in `DEFINITIONS_REGISTRY` under `factor_tilts`. Add `compute_factor_tilts()` in `src/analytics/factors.py`
- [ ] **1.5 — PDF report endpoint**: Wire existing `generate_html_report()` in [streamlit_export.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/streamlit_export.py) to weasyprint. Add `GET /api/reports/{run_id}/pdf` in [server.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/api/server.py)
- [ ] **1.6 — Golden tests for new metrics**: Add test scenarios for HHI, VaR comparison, and factor tilts

### Already done (skip):
- ✅ Functional refactor (10 pure analytics modules)
- ✅ Attribution (Brinson with Carino linking)
- ✅ Risk contributions with per-asset VaR decomposition
- ✅ FRED macro data integration

---

## Phase 2: Auth & Multi-Tenancy *(~5-7 days)*

> **Status: ~20% done.** DB schema has `user_id` columns. `SupabaseRepo` exists but hardcodes `user_id=1`.  
> **This is the biggest remaining lift.**

### Backend auth:

- [ ] **2.1 — JWT middleware**: Add FastAPI dependency `get_current_user(request)` that extracts and validates Supabase JWT from `Authorization: Bearer <token>` header. Add to [server.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/src/api/server.py)
- [ ] **2.2 — Wire user_id through repo**: Update [storage_supabase/repo.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/storage_supabase/repo.py) — remove hardcoded `user_id=1`, accept `user_id` parameter from auth context. Key methods: `get_default_portfolio_id()`, `replace_trades()`, `get_trades()`, `create_run()`, etc.
- [ ] **2.3 — RLS policies**: Add Row Level Security in Supabase dashboard:
  - `portfolios` → `user_id = auth.uid()`
  - `transactions` → via portfolio's `user_id`
  - `runs` → via portfolio's `user_id`
  - `run_artifacts` → via run's portfolio's `user_id`

### Frontend auth:

- [ ] **2.4 — Supabase Auth SDK**: Add `@supabase/ssr` to `web/package.json`. Configure Supabase client with project URL and anon key
- [ ] **2.5 — Login/signup pages**: Create `/login` route with email/password + (optionally) OAuth (Google). Use Supabase's built-in auth UI or minimal custom form
- [ ] **2.6 — Auth middleware**: Protect all `/app/(routes)/*` pages. Redirect unauthenticated users to `/login`
- [ ] **2.7 — Pass JWT to API**: Update [api.ts](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/web/src/lib/api.ts) `fetchJson()` to include `Authorization` header from Supabase session

### Data isolation:

- [ ] **2.8 — Kill hybrid mode**: Set production to `STORAGE_MODE=supabase`. SQLite only for `STORAGE_MODE=local` (dev/tests)
- [ ] **2.9 — Supabase models audit**: Verify [storage_supabase/models.py](file:///Users/donatdermaku/PycharmProjects/personal-investor-assistant/storage_supabase/models.py) has `user_id` on all user-facing tables, add where missing

### Already done (skip):
- ✅ `User`, `Portfolio`, `WatchlistItem`, `AppSettings` models have `user_id` columns
- ✅ `SupabaseRepo` has full CRUD for trades, snapshots, runs, artifacts
- ✅ Supabase Storage configured for artifact upload/download
- ✅ Market data persistent cache to Supabase Storage

---

## Phase 3: Beta Polish *(~3-5 days)*

> **Status: Not started.** This is the final sprint before inviting users.

- [ ] **3.1 — Landing page**: Simple public page explaining what Nexus does, with "Sign up for Beta" CTA
- [ ] **3.2 — Onboarding flow**: First-time user experience — create portfolio → upload CSV or try demo → see results
- [ ] **3.3 — Error monitoring**: Structured logs already exist via `structlog`. Add Cloud Run log query for error rate dashboard
- [ ] **3.4 — CORS for production**: Update `NEXUS_ALLOWED_ORIGINS` in Cloud Run to match actual Vercel deployment URL
- [ ] **3.5 — Rate limiting per user**: Current rate limiter uses IP. Switch to user-based limiting once auth is in place
- [ ] **3.6 — 50-user smoke test**: Create 50 test accounts, each with different portfolio sizes. Verify isolation, performance, and no data leaks

---

## Verification Plan

### Automated (on every PR)
```bash
# Backend
pytest tests/ -q                    # 120 tests including golden regression
python -m mypy src/ --config mypy.ini

# Frontend
npm --prefix web run lint
npm --prefix web run test
npm --prefix web run build
```

### Manual (before beta launch)
- [ ] Create 2 accounts, verify complete data isolation
- [ ] Upload CSV on Account A, verify Account B cannot see it
- [ ] Test concurrent run creation from 2 users
- [ ] Verify PDF report generation end-to-end
- [ ] Test login/logout/session expiry flows

---

## Effort Estimate

| Phase | Days | Blocked By |
|-------|------|------------|
| Phase 0 | 1-2 | Nothing |
| Phase 1 | 3-5 | Nothing (parallel with Phase 0) |
| Phase 2 | 5-7 | Phase 0 |
| Phase 3 | 3-5 | Phase 2 |
| **Total** | **~12-19 days** | |

---

## What's Explicitly NOT in Scope (Zero-Cost Beta)

- ❌ Async workers / Celery / task queues
- ❌ Premium data providers (Bloomberg, Refinitiv)
- ❌ Billing / Stripe integration
- ❌ Mobile app
- ❌ Real-time market data streaming
- ❌ Data Provider ABC interface (current concrete implementations work fine)
- ❌ Pydantic InputContract (current validation is sufficient)
- ❌ Complete Streamlit removal (it works, it's not user-facing, deprioritize)
