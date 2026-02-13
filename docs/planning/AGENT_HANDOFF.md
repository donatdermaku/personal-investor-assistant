# Agent Handoff Context

> **Date:** 2026-02-13
> **Status:** Phase 0, Phase 1, and Phase 2 completed. Phase 3 implementation completed except smoke test execution.

## Current State
We completed Phase 0 cleanup, completed Phase 1, advanced Phase 2 auth/multi-tenancy, and started Phase 3 beta polish:
- Fixed duplicate `options` block in `cloudbuild.yaml`.
- Added `.gitignore` rules for loose root CSV patterns.
- Simplified `storage/datamanager.py` from `db/files/hybrid` to `local/supabase` with strict mode separation.
- Added GitHub Actions CI workflow for backend (`pytest`) and frontend (`lint + build`) on push/PR.
- Added `src/analytics/metrics_registry.py` and gated `/api/v1/definitions` to registered metrics only.
- Added HHI concentration analytics in `src/analytics/concentration.py`, including thresholds and API/export wiring (`concentration_summary.json`).
- Added VaR budget comparison in `src/analytics/risk.py` and wired it into benchmark comparison summary payloads returned by API/report artifacts.
- Added Factor Tilt analytics in `src/analytics/factors.py` and wired `factor_tilts.json` into exports and API payload (`get_run` + export route).
- Added PDF report rendering with WeasyPrint and endpoint `GET /api/reports/{run_id}/pdf` (plus `/api/v1/reports/{run_id}/pdf` alias).
- Added golden verification coverage for new value-engine outputs (HHI concentration, factor tilts, VaR budget comparison) in `tests/test_golden_value_engine.py`.
- Added backend auth dependency `get_current_user` in API (`src/api/server.py`) with Supabase bearer token parsing and HS256 JWT verification against `SUPABASE_JWT_SECRET`.
- Added explicit JWT header algorithm guard (`alg` must be `HS256`) to avoid accidental acceptance under mismatched signing config.
- Updated default portfolio resolution and run/portfolio endpoints to use authenticated `user_id` in Supabase mode.
- Updated Supabase repo to remove hardcoded `user_id=1` and scope default portfolio lookup/creation by provided `user_id` (`storage_supabase/repo.py`), with `Portfolio.user_id` modeled in `storage_supabase/models.py`.
- Extended Supabase scoping to run and artifact reads:
  - `list_runs`, `get_latest_run`, `get_run_by_id`, and `get_artifact_bytes` now accept/propagate `user_id` context.
  - API run/read/export/report endpoints now enforce run ownership in Supabase mode before returning data.
- Added frontend auth wiring:
  - `@supabase/ssr` and `@supabase/supabase-js` installed in `web/package.json`
  - `/login` page created (`web/src/app/login/page.tsx`)
  - route-protecting middleware added (`web/src/middleware.ts`)
  - bearer token pass-through added in web API client (`web/src/lib/api.ts`)
- Added Supabase RLS policy script for manual application in dashboard SQL editor: `docs/planning/supabase_rls_policies.sql`.
- Confirmed RLS enabled for `portfolios`, `transactions`, `runs`, and `run_artifacts` in Supabase production project.
- Updated deployment defaults to Supabase mode:
  - `render.yaml` now uses `STORAGE_MODE=supabase` and includes required Supabase env var placeholders.
  - `docs/DEPLOYMENT.md` now documents Supabase production auth/database env vars.
- Hardened service-context fallback behavior:
  - Supabase service user fallback now uses explicit `SUPABASE_SERVICE_CONTEXT_USER_ID` (legacy `SUPABASE_DEFAULT_USER_ID` still read but deprecated).
  - Request paths in Supabase mode remain JWT-required (`401` when auth context is missing).
- Added backend API tests for new multi-tenant behavior in `tests/test_api_server.py`:
  - Bearer required in Supabase mode
  - User-scoped run listing propagation
  - Run ownership denial path
  - User-scoped artifact export propagation
- Added Phase 3 landing/onboarding flow:
  - Public landing page with beta signup UX at `/` (`web/src/app/page.tsx`)
  - Protected onboarding route at `/onboarding` with portfolio creation + CSV upload flow (`web/src/app/onboarding/page.tsx`)
  - New backend endpoint `POST /api/v1/portfolios` for creating user portfolios (`src/api/server.py`)
  - Frontend API wiring for portfolio creation (`web/src/lib/api.ts`)
  - CSV onboarding validation utilities + unit tests (`web/src/lib/onboarding.ts`, `web/src/lib/__tests__/onboarding.test.ts`)
  - Middleware protection extended to `/onboarding` (`web/src/middleware.ts`)
  - Backend tests for new portfolio endpoint auth and success path (`tests/test_api_server.py`)
- Added remaining Phase 3 hardening (pre-smoke-test):
  - User-based rate limiting in API middleware (falls back to IP when user cannot be derived) (`src/api/server.py`)
  - Production-aware CORS resolution (`NEXUS_ENV=production` + empty-default allowlist when not explicitly configured) (`src/api/server.py`)
  - Cloud Run error monitoring primitives:
    - structured `ERROR_EVENT` logging for unhandled exceptions and HTTP 5xx
    - admin inspection endpoint `GET /admin/error-events`
    - runbook doc with log queries (`docs/CLOUD_RUN_ERROR_MONITORING.md`)
  - Deployment docs/config updated for `NEXUS_ENV=production` (`docs/DEPLOYMENT.md`, `render.yaml`)
- Production hotfixes after deploy validation:
  - Docker image fixed to include `storage_supabase` package (resolved Cloud Run `ModuleNotFoundError: storage_supabase` startup crash).
  - `/ops/health` fixed to support both run model field variants (`id` and `run_id`) to prevent 500 errors in mixed local/supabase model paths (`src/api/server.py`).
  - Added regression test coverage for `/ops/health` latest-run ID fallback (`tests/test_api_server.py`).

## Immediate Next Actions
The next agent should focus on remaining **Phase 2 + Phase 3** operational work:
1. Run smoke test plan for 50 beta users against current Cloud Run revision.
2. If smoke passes, monitor `/admin/error-events` and Cloud Run logs during initial onboarding window.

## Key Architectural Decisions
- **Auth:** We will use Supabase Auth (JWT) + RLS. No custom auth system.
- **Database:** `STORAGE_MODE=supabase` for production. `local` for dev. No `hybrid`.
- **Market Data:** Stays in local cache + Supabase Storage backup (Parquet). No DB for prices.

## References
- [Master Task List](file:///Users/donatdermaku/.gemini/antigravity/brain/d8a81cb5-6cfc-4e86-ada2-8ac083bfd420/task.md)
- [Implementation Plan](file:///Users/donatdermaku/.gemini/antigravity/brain/d8a81cb5-6cfc-4e86-ada2-8ac083bfd420/implementation_plan.md)
- [Gap Analysis](file:///Users/donatdermaku/.gemini/antigravity/brain/d8a81cb5-6cfc-4e86-ada2-8ac083bfd420/gap_analysis.md)
