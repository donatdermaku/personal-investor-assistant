# Supabase Setup

Required environment variables (backend only):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `SUPABASE_DB_URL`
- `SUPABASE_STORAGE_BUCKET` (optional, default `nexus-artifacts`)

## Steps

1. Create a Supabase project.
2. Copy the Postgres connection string into `SUPABASE_DB_URL`.
3. Copy the service role key into `SUPABASE_SERVICE_ROLE_KEY`.
4. Set `SUPABASE_URL` to the project URL.
5. Create a storage bucket named `nexus-artifacts` (or set `SUPABASE_STORAGE_BUCKET`).

## Migrations

Run Alembic against Supabase:

```bash
export SUPABASE_DB_URL="postgresql://..."
alembic upgrade head
```

## Render

Set the env vars in Render and redeploy. If the vars are missing, the app falls back to local SQLite/files.
