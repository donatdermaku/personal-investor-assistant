-- Phase 2: Supabase RLS policies for user data isolation
-- Apply in Supabase SQL editor after confirming table/column names in target environment.

begin;

alter table if exists public.portfolios enable row level security;
alter table if exists public.transactions enable row level security;
alter table if exists public.runs enable row level security;
alter table if exists public.run_artifacts enable row level security;

drop policy if exists portfolios_select_own on public.portfolios;
drop policy if exists portfolios_insert_own on public.portfolios;
drop policy if exists portfolios_update_own on public.portfolios;
drop policy if exists portfolios_delete_own on public.portfolios;

create policy portfolios_select_own
on public.portfolios
for select
using (user_id = auth.uid()::text);

create policy portfolios_insert_own
on public.portfolios
for insert
with check (user_id = auth.uid()::text);

create policy portfolios_update_own
on public.portfolios
for update
using (user_id = auth.uid()::text)
with check (user_id = auth.uid()::text);

create policy portfolios_delete_own
on public.portfolios
for delete
using (user_id = auth.uid()::text);

drop policy if exists transactions_select_own on public.transactions;
drop policy if exists transactions_insert_own on public.transactions;
drop policy if exists transactions_update_own on public.transactions;
drop policy if exists transactions_delete_own on public.transactions;

create policy transactions_select_own
on public.transactions
for select
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = transactions.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy transactions_insert_own
on public.transactions
for insert
with check (
  exists (
    select 1
    from public.portfolios p
    where p.id = transactions.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy transactions_update_own
on public.transactions
for update
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = transactions.portfolio_id
      and p.user_id = auth.uid()::text
  )
)
with check (
  exists (
    select 1
    from public.portfolios p
    where p.id = transactions.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy transactions_delete_own
on public.transactions
for delete
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = transactions.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

drop policy if exists runs_select_own on public.runs;
drop policy if exists runs_insert_own on public.runs;
drop policy if exists runs_update_own on public.runs;
drop policy if exists runs_delete_own on public.runs;

create policy runs_select_own
on public.runs
for select
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = runs.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy runs_insert_own
on public.runs
for insert
with check (
  exists (
    select 1
    from public.portfolios p
    where p.id = runs.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy runs_update_own
on public.runs
for update
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = runs.portfolio_id
      and p.user_id = auth.uid()::text
  )
)
with check (
  exists (
    select 1
    from public.portfolios p
    where p.id = runs.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

create policy runs_delete_own
on public.runs
for delete
using (
  exists (
    select 1
    from public.portfolios p
    where p.id = runs.portfolio_id
      and p.user_id = auth.uid()::text
  )
);

drop policy if exists run_artifacts_select_own on public.run_artifacts;
drop policy if exists run_artifacts_insert_own on public.run_artifacts;
drop policy if exists run_artifacts_update_own on public.run_artifacts;
drop policy if exists run_artifacts_delete_own on public.run_artifacts;

create policy run_artifacts_select_own
on public.run_artifacts
for select
using (
  exists (
    select 1
    from public.runs r
    join public.portfolios p on p.id = r.portfolio_id
    where r.id = run_artifacts.run_id
      and p.user_id = auth.uid()::text
  )
);

create policy run_artifacts_insert_own
on public.run_artifacts
for insert
with check (
  exists (
    select 1
    from public.runs r
    join public.portfolios p on p.id = r.portfolio_id
    where r.id = run_artifacts.run_id
      and p.user_id = auth.uid()::text
  )
);

create policy run_artifacts_update_own
on public.run_artifacts
for update
using (
  exists (
    select 1
    from public.runs r
    join public.portfolios p on p.id = r.portfolio_id
    where r.id = run_artifacts.run_id
      and p.user_id = auth.uid()::text
  )
)
with check (
  exists (
    select 1
    from public.runs r
    join public.portfolios p on p.id = r.portfolio_id
    where r.id = run_artifacts.run_id
      and p.user_id = auth.uid()::text
  )
);

create policy run_artifacts_delete_own
on public.run_artifacts
for delete
using (
  exists (
    select 1
    from public.runs r
    join public.portfolios p on p.id = r.portfolio_id
    where r.id = run_artifacts.run_id
      and p.user_id = auth.uid()::text
  )
);

commit;
