# Deployment Guide (Render + Vercel)

This guide covers the free-tier deployment for the Nexus Analytics Platform backend (Render) and frontend (Vercel).

## Backend (Render)

### Service setup
- Service type: Web Service
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn src.api.server:app --host 0.0.0.0 --port $PORT`

### Required environment variables
- `NEXUS_DB_PATH=/var/data/user.db`
- `NEXUS_EXPORT_DIR=/var/data/exports`
- `NEXUS_ALLOWED_ORIGINS=https://<your-vercel-app>.vercel.app`
- `STORAGE_MODE=supabase`
- `SUPABASE_DB_URL=postgresql://...`
- `SUPABASE_URL=https://<project>.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY=<service-role-key>`
- `SUPABASE_JWT_SECRET=<jwt-secret>`
- `SUPABASE_SERVICE_CONTEXT_USER_ID=<internal-only-user-id>` (optional; only for non-HTTP/background service context)

Security note:
- In `STORAGE_MODE=supabase`, HTTP requests must present a valid bearer token; there is no request-path fallback user.

### Persistent disk
- In Render, attach a persistent disk and mount it at `/var/data`.
- The exports and local cache will be stored under that mount.

### Verify backend
```bash
curl https://<your-render-service>.onrender.com/health
curl https://<your-render-service>.onrender.com/latest-run
```

Expected behavior:
- `/health` returns `{ "status": "ok" }`.
- `/latest-run` returns the latest manifest or a 404 if no runs exist yet.

## Frontend (Vercel)

### Project setup
- Import the repo in Vercel.
- The `vercel.json` sets the root to `/web`.

### Required environment variables
- `NEXT_PUBLIC_API_URL=https://<your-render-service>.onrender.com`

### Verify frontend
- Visit `https://<your-vercel-app>.vercel.app/overview`.
- Ensure data loads and charts render.

## Export Endpoints

These serve existing artifacts only; they do not compute new data:
- `/run/{run_id}/export/summary-json`
- `/run/{run_id}/export/performance-csv`
- `/run/{run_id}/export/monthly-returns-csv`

## Common failure cases

- CORS errors in the browser:
  - Confirm `NEXUS_ALLOWED_ORIGINS` matches the exact Vercel URL with `https://`.
- `No runs found`:
  - Run the compute pipeline to generate a run and artifacts.
- `Artifact not found`:
  - Ensure exports exist under `NEXUS_EXPORT_DIR/<run_id>/`.
