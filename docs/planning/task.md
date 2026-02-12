# Project Nexus: Master Task List

> **Goal:** Validated "Zero-Cost" Beta Launch (50 users)  
> **Constraint:** $0 incremental spend. Supabase-first architecture.  
> **Status:** Phase 0 (Cleanup) completed. Phase 1 (Value Engine) in progress.

## Phase 0: Foundation Cleanup
- [x] Golden Test Harness (7 scenarios implemented)
- [x] yfinance Isolation (Rate limiting, retry, caching implemented)
- [x] CSV Input Validation (`validate_ledger()` implemented)
- [x] Fix `cloudbuild.yaml` duplicate options block
- [x] Clean repo root (move/ignore loose CSVs)
- [x] Simplify `STORAGE_MODE` to `local` vs `supabase` (remove `hybrid`)
- [x] Setup GitHub Actions CI (pytest + npm build on push)

## Phase 1: Value Engine ("Killer Insights")
- [x] **Metrics Registry Enforcement**: Create `src/analytics/metrics_registry.py` to gate API exposure
  - [x] Define registry structure
  - [x] Wire to `DEFINITIONS_REGISTRY`
  - [x] Add enforcement decorator/check
- [x] **HHI Concentration Index**: Implement `src/analytics/concentration.py`
  - [x] Pure function `compute_hhi(weights)`
  - [x] Add thresholds (Diversified <0.15, etc.)
  - [x] Register metric
- [x] **VaR Budget Comparison**: Update `src/analytics/risk.py`
  - [x] Implement `compare_var_budget(portfolio, benchmark, alpha)`
  - [x] Add to API/Report
- [x] **Factor Tilt Computation**: Implement `src/analytics/factors.py`
  - [x] `compute_factor_tilts()` based on definitions
- [x] **PDF Report Endpoint**:
  - [x] Wire `weasyprint` to `generate_html_report`
  - [x] Create `GET /api/reports/{id}/pdf` endpoint
- [x] **Verification**: Add golden tests for new metrics

## Phase 2: Auth & Multi-Tenancy (Supabase-First)
- [ ] **Backend Auth (FastAPI)**:
  - [ ] Add `get_current_user` dependency (JWT validation)
  - [ ] Extract user_id from Supabase token
- [ ] **Repo Layer Update**:
  - [ ] Update `SupabaseRepo` to accept `user_id` context
  - [ ] Remove hardcoded `user_id=1`
- [ ] **Database RLS**:
  - [ ] Apply RLS policies to `portfolios`, `transactions`, `runs`
- [ ] **Frontend Auth (Next.js)**:
  - [ ] Install `@supabase/ssr`
  - [ ] Create `/login` page
  - [ ] Add Auth middleware for protected routes
  - [ ] Pass JWT in API calls
- [ ] **Data Isolation**:
  - [ ] Audit all Supabase models for `user_id`
  - [ ] Switch production config to `STORAGE_MODE=supabase`

## Phase 3: Beta Polish
- [ ] Landing Page (Simple "Sign up for Beta")
- [ ] Onboarding Flow (Portfolio creation + CSV upload)
- [ ] Cloud Run Error Monitoring (Log queries)
- [ ] Production CORS Update
- [ ] User-based Rate Limiting
- [ ] Smoke Test (50 users)

## Reference Documents
- [Implementation Plan](file:///Users/donatdermaku/.gemini/antigravity/brain/d8a81cb5-6cfc-4e86-ada2-8ac083bfd420/implementation_plan.md) - Detailed breakdown of phases and architecture
- [Gap Analysis](file:///Users/donatdermaku/.gemini/antigravity/brain/d8a81cb5-6cfc-4e86-ada2-8ac083bfd420/gap_analysis.md) - Context on current codebase state vs plan
