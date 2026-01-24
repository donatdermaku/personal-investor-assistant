# Persistence Architecture

Nexus supports two persistence modes:

1) Local (default)
- SQLite + local artifacts directory
- Used when Supabase env vars are missing

2) Supabase
- Postgres for portfolios, transactions, runs, artifact metadata
- Supabase Storage for artifacts
- Enabled when:
  - `SUPABASE_DB_URL`
  - `SUPABASE_URL`
  - `SUPABASE_SERVICE_ROLE_KEY`
  are present

## Endpoint behavior

Endpoints remain stable across modes:
- `/run`, `/runs`, `/latest-run`, `/run/{id}`, `/portfolio/{id}`, `/definitions`, `/run/{id}/export/{artifact}`

The repo layer selects the backend automatically.
