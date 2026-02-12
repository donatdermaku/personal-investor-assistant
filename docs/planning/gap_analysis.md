# Gap Analysis: Project Nexus Plan vs Actual Codebase

> **Generated:** 2026-02-12 by code-level audit  
> **Scope:** Every item in the "Agent Execution Guide & Roadmap" checked against actual source code

---

## Overview Table

| Plan Item | Plan Says | What Actually Exists | Status |
|-----------|-----------|---------------------|--------|
| **Golden Test Harness** (0.1) | Create `tests/golden/` with 5 static CSVs and expected JSON | 7 golden scenarios exist: `single_trade`, `multi_asset`, `cash_only`, `dividends_only`, `no_trades`, `scenario_A`, `scenario_B`. Full regression tests in `test_golden_portfolios.py` + `test_golden_portfolios_extended.py` | ✅ Done |
| **yfinance Isolation** (0.2) | Isolate all yfinance calls in `market_data/` | All Yahoo calls in `market_data/yahoo.py`. Rate limiter with backoff/retry in `market_data/rate_limiter.py`. Persistent cache in `market_data/persistent_cache.py` | ✅ Done |
| **InputContract** (0.2) | Pydantic model for CSV uploads | `validate_ledger()` in `src/portfolio.py` checks required columns (`date`, `ticker`, `action`, `quantity`, `price`). `docs/INPUT_CONTRACT.md` documents the schema. **No Pydantic model** — validation is manual pandas checks | ⚠️ Partial |
| **No Silent Math Rule** | `src/analytics/metrics_registry.py` with enforcement | `src/definitions.py` has `DEFINITIONS_REGISTRY` dict with ~25 metrics, each having `title`, `definition_md`, `assumptions`, `warnings`. **But no enforcement mechanism** — metrics can exist without registration. No `metrics_registry.py` file | ⚠️ Partial |
| **Functional Refactor** (1.1) | Pure functions in `src/analytics/`, e.g. `calculate_drawdown(series) -> float` | `src/analytics/` has 10 pure-function modules: `risk.py`, `attribution.py`, `correlation.py`, `comparative.py`, `macro.py`, `rolling.py`, `streaming.py`, `contracts.py`, `required_start.py`. `compute_drawdown()` exists in `src/portfolio.py` as a pure function. Attribution uses Brinson decomposition with Carino linking — already pure | ✅ Done |
| **Attribution (Brinson-Fachler)** (1.1) | Must be pure function | `src/analytics/attribution.py` — 213 lines, pure functional. Takes DataFrames in, returns `AttributionOutput` dataclass. Uses `compute_attribution()` with allocation/selection/interaction decomposition. No DB/API calls | ✅ Done |
| **Risk Budget / VaR Comparison** (1.2) | Compare User VaR vs Benchmark VaR | `src/analytics/risk.py` computes VaR inside `compute_risk_contributions()` (per-asset VaR decomposition, `var_alpha=0.05`). **But no standalone VaR comparison function** — no user-vs-benchmark VaR report | ⚠️ Partial — VaR exists, comparison missing |
| **HHI Concentration** (1.2) | Calculate Herfindahl-Hirschman Index | **Not implemented.** No HHI computation anywhere in the codebase (searched `src/` for `HHI`, `herfindahl`, `concentration`) | ❌ Missing |
| **Factor Tilt** (1.2) | Correlation to Momentum/Value factors | Defined in `DEFINITIONS_REGISTRY` (`factor_tilts` entry) but **no compute function exists**. Searched `src/analytics/` for `factor_tilt`, `momentum`, `value_factor` — nothing found | ❌ Missing |
| **PDF Report Generation** (1.3) | `GET /api/reports/{run_id}/pdf` with Jinja2 + weasyprint | `weasyprint` is in `requirements.txt`. `src/streamlit_export.py` has `generate_html_report()` (107 lines, Jinja2 templating). **But no PDF endpoint exists** in `server.py`. HTML report is generated but not served as PDF | ⚠️ Partial — HTML exists, PDF endpoint missing |
| **Data Provider Interface** (2.1) | `class MarketDataProvider(ABC)` | **No ABC interface.** `market_data/yahoo.py` and `market_data/fred.py` are concrete implementations. `market_data/store.py` is the coordinator. No abstract provider pattern | ❌ Missing |
| **FRED Macro Data** (2.2) | DGS10, CPIAUCSL, T10Y2Y from FRED | `market_data/fred.py` fetches FRED data. `src/analytics/macro.py` computes regime signals, yield curve analysis, CPI/inflation context. Fully integrated into pipeline | ✅ Done |
| **"Real Return" (Nominal - Inflation)** (2.2) | Show in UI | `src/analytics/macro.py` computes macro context. Exposed via API. Frontend `ContextPanel.tsx` shows macro data. Unclear if specific "Real Return" metric is displayed | ⚠️ Needs verification |
| **Supabase Auth** (3.1) | `@supabase/auth-helpers-nextjs` on frontend, JWT validation on backend | **Zero auth code.** No JWT validation in FastAPI. No `get_current_user` dependency. No `@supabase/auth-helpers-nextjs` in `package.json`. No login/signup UI | ❌ Missing |
| **User Isolation / Multi-tenancy** (3.2) | `user_id` on ALL tables, RLS | DB models already have `user_id` on: `Portfolio`, `WatchlistItem`, `AppSettings`. `Run` and `Trade` link through `portfolio_id → Portfolio.user_id`. `repo.py` has `get_user_id()`, `get_default_portfolio_id(user_id)`, `list_watch_tickers(user_id)`. **But `server.py` never passes authenticated user_id** — hardcoded single-user flow | ⚠️ Partial — schema ready, auth missing |
| **GitHub Actions CI** (5) | Run `make test` on push | `.github/workflows/nigthly.yml` exists — runs nightly data ingest + report build + GitHub Pages deploy. **But no CI on push/PR** (no test runner, no lint, no golden test gate) | ⚠️ Partial — nightly exists, CI missing |
| **Env Vars** (5) | `NEXUS_ENV`, `STORAGE_MODE`, `MARKET_DATA_PROVIDER` | `STORAGE_MODE` used in codebase. `NEXUS_ENV` not found. `MARKET_DATA_PROVIDER` not found — Yahoo is hardcoded. Many other env vars exist: `NEXUS_DB_PATH`, `NEXUS_EXPORT_DIR`, `NEXUS_ALLOWED_ORIGINS`, `NEXUS_RATE_LIMIT_*` | ⚠️ Partial |

---

## Summary by Phase

### Phase 0: Scope Lock & Foundation
| Task | Status | What's Left |
|------|--------|-------------|
| Golden Harness | ✅ Complete | 7 scenarios, regression tests passing |
| yfinance Isolation | ✅ Complete | Rate limiting, retry, caching all done |
| InputContract (Pydantic) | ⚠️ Partial | Validation exists but no Pydantic model — decide if this is actually needed given current validation works |

**Verdict: Phase 0 is ~90% done.** Only missing a formal Pydantic InputContract, which may be optional.

---

### Phase 1: Value Engine Hardening
| Task | Status | What's Left |
|------|--------|-------------|
| Functional Refactor | ✅ Complete | 10 pure analytics modules |
| No Silent Math / Metrics Registry | ⚠️ Partial | Registry dict exists with 25 metrics. Missing: enforcement mechanism (`metrics_registry.py` that gates API/report exposure) |
| VaR Budget (User vs Benchmark) | ⚠️ Partial | VaR computation exists. Missing: standalone comparison function |
| HHI Concentration | ❌ Missing | Not implemented anywhere |
| Factor Tilt | ❌ Missing | Defined in registry, no compute function |
| PDF Report Endpoint | ⚠️ Partial | HTML report via Jinja2 exists. Missing: PDF endpoint |

**Verdict: Phase 1 is ~50% done.** Core analytics refactor is complete. Missing the "killer insights" (HHI, Factor Tilt, VaR comparison) and the PDF endpoint.

---

### Phase 2: Free Data & Context
| Task | Status | What's Left |
|------|--------|-------------|
| Data Provider ABC | ❌ Missing | No abstract interface — concrete implementations only |
| FRED Macro Data | ✅ Complete | DGS10, CPI, T10Y2Y fully integrated |
| Real Return metric | ⚠️ Verify | Macro analytics exist, need to verify specific metric |

**Verdict: Phase 2 is ~70% done.** The data itself is all there. Missing the formal ABC pattern (which is arguably a refactor, not a feature).

---

### Phase 3: Beta Readiness
| Task | Status | What's Left |
|------|--------|-------------|
| Supabase Auth | ❌ Missing | Zero auth code on frontend or backend |
| User Isolation | ⚠️ Partial | DB schema has `user_id` columns. `repo.py` has user-aware queries. Missing: auth middleware to supply the actual user_id |
| CI on Push/PR | ❌ Missing | Only nightly workflow exists |

**Verdict: Phase 3 is ~20% done.** The DB schema is multi-tenant-ready, but all the auth and CI work is unbuilt.

---

## Actual Remaining Work (Prioritized)

### High Priority (blocks beta launch)
1. **Auth layer** — Supabase Auth (frontend) + JWT middleware (backend) + wire `user_id` flow
2. **CI on push** — GitHub Action running `make test` + `npm run build` on every PR
3. **Metrics Registry enforcement** — `metrics_registry.py` that gates unregistered metrics

### Medium Priority (differentiators)
4. **HHI Concentration metric** — pure function in `src/analytics/`
5. **VaR Budget comparison** — standalone function comparing portfolio vs benchmark VaR
6. **Factor Tilt computation** — implement what's already defined in registry
7. **PDF report endpoint** — wire existing HTML generation to weasyprint

### Low Priority (nice-to-have)
8. **Data Provider ABC** — formalize the interface (current code works fine without it)
9. **Pydantic InputContract** — current validation is functional
10. **`NEXUS_ENV` / `MARKET_DATA_PROVIDER` env vars** — partially exists
