-- ===========================================================================
-- FPL Agent - Supabase (Postgres) schema
-- ===========================================================================
-- Run this in the Supabase SQL editor (Dashboard -> SQL -> New query).
-- Free tier is plenty: ~500MB Postgres + built-in auth.
--
-- Design:
--   * auth.users is managed by Supabase Auth (magic-link email login).
--   * profiles      - one row per user: links their public FPL team id.
--   * player_gw     - historical per-player per-gameweek stats (all seasons).
--   * live_players  - latest snapshot of the current season's players.
--   * live_fixtures - latest snapshot of fixtures + official difficulty.
--   * news          - crawled, reliability-scored news items.
--   * recommendations - per-user log of recommendations made.
-- Row Level Security: user tables are private per-user; stats/news tables are
-- world-readable (they're public data) and writable only by the service role
-- (the GitHub Actions crawler uses the service key).
-- ===========================================================================

-- ---------- users ----------------------------------------------------------
create table if not exists public.profiles (
  id uuid primary key references auth.users (id) on delete cascade,
  fpl_team_id integer,
  display_name text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

alter table public.profiles enable row level security;

create policy "read own profile"   on public.profiles for select using (auth.uid() = id);
create policy "insert own profile" on public.profiles for insert with check (auth.uid() = id);
create policy "update own profile" on public.profiles for update using (auth.uid() = id);

-- auto-create a profile on signup
create or replace function public.handle_new_user()
returns trigger language plpgsql security definer set search_path = public as $$
begin
  insert into public.profiles (id) values (new.id) on conflict do nothing;
  return new;
end $$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- ---------- historical stats ------------------------------------------------
create table if not exists public.player_gw (
  id bigint generated always as identity primary key,
  season text not null,
  gameweek smallint not null,
  name text not null,
  position text,
  team text,
  opponent_team smallint,
  was_home boolean,
  value real,
  minutes smallint,
  total_points smallint,
  form_points real,
  form_minutes real,
  form_ict real,
  last_gw_points real,
  season_form_points real,
  opp_goals_conceded_form real,
  opp_goals_scored_form real,
  fixture_difficulty real
);
create index if not exists idx_pgw_season_gw on public.player_gw (season, gameweek);
create index if not exists idx_pgw_name on public.player_gw (name);

-- ---------- live snapshot ----------------------------------------------------
create table if not exists public.live_players (
  player_id integer primary key,
  fetched_at timestamptz not null default now(),
  name text,
  web_name text,
  team text,
  team_code integer,
  position text,
  price real,
  form real,
  ppg real,
  ep_next real,
  status text,
  chance smallint,
  news text
);

create table if not exists public.live_fixtures (
  fixture_id integer primary key,
  fetched_at timestamptz not null default now(),
  event smallint,
  team_h smallint,
  team_a smallint,
  team_h_difficulty smallint,
  team_a_difficulty smallint,
  finished boolean,
  kickoff timestamptz
);

-- ---------- news -------------------------------------------------------------
create table if not exists public.news (
  id bigint generated always as identity primary key,
  fetched_at timestamptz not null default now(),
  season text,
  gameweek smallint,
  player text,
  source text,
  reliability real,
  corroboration smallint,
  category text,
  recency text,
  headline text,
  detail text
);
create index if not exists idx_news_player on public.news (player);
create index if not exists idx_news_fetched on public.news (fetched_at desc);

-- ---------- recommendations ---------------------------------------------------
create table if not exists public.recommendations (
  id bigint generated always as identity primary key,
  user_id uuid references auth.users (id) on delete set null,
  created_at timestamptz not null default now(),
  decision text,
  mode text,
  source text,
  season text,
  gameweek smallint,
  formula text,
  request jsonb,
  result jsonb
);
create index if not exists idx_rec_user on public.recommendations (user_id, created_at desc);

-- ---------- row level security -------------------------------------------------
-- Public data: anyone (incl. anonymous) can read; only service role writes.
alter table public.player_gw     enable row level security;
alter table public.live_players  enable row level security;
alter table public.live_fixtures enable row level security;
alter table public.news          enable row level security;

create policy "public read player_gw"     on public.player_gw     for select using (true);
create policy "public read live_players"  on public.live_players  for select using (true);
create policy "public read live_fixtures" on public.live_fixtures for select using (true);
create policy "public read news"          on public.news          for select using (true);
-- (no insert/update policies for anon/authenticated -> only service_role writes)

-- Per-user data
alter table public.recommendations enable row level security;
create policy "read own recs"   on public.recommendations for select using (auth.uid() = user_id);
create policy "insert own recs" on public.recommendations for insert with check (auth.uid() = user_id);
