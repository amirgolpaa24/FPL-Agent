"""
Scheduled news + data crawler. Designed to run on GitHub Actions (free cron)
in the run-up to each gameweek deadline:

  1. Fetches fresh live FPL data (players, prices, injuries, fixtures).
  2. Gathers performance-relevant news (FPL official flags + BBC/Guardian/Sky
     RSS + optional NewsAPI), scored by reliability/corroboration/recency.
  3. Upserts everything to Supabase Postgres (if SUPABASE_URL +
     SUPABASE_SERVICE_KEY env vars are set) so every user/app instance reads
     the same fresh data. Falls back to local files/SQLite otherwise.

Intensity scales with deadline proximity: within DEADLINE_WINDOW_HOURS of the
next kickoff deadline it always crawls; otherwise it still refreshes data but
you can schedule it less often. Exit code 0 even on partial failure - a broken
feed must never kill the scheduled job.

Env:
  SUPABASE_URL          e.g. https://abcd.supabase.co     (optional)
  SUPABASE_SERVICE_KEY  service-role key (secret!)         (optional)
  NEWS_API_KEY          NewsAPI key                        (optional)
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEADLINE_WINDOW_HOURS = 48


def hours_to_deadline(live) -> float | None:
    """Hours until the next unfinished fixture's kickoff (proxy for deadline)."""
    upcoming = [f.get("kickoff_time") for f in live.fixtures
                if not f.get("finished") and f.get("kickoff_time")]
    if not upcoming:
        return None
    nxt = min(upcoming)
    try:
        when = dt.datetime.fromisoformat(nxt.replace("Z", "+00:00"))
        return (when - dt.datetime.now(dt.timezone.utc)).total_seconds() / 3600
    except ValueError:
        return None


def upsert_supabase(table: str, rows: list[dict], on_conflict: str | None = None) -> bool:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")
    if not (url and key and rows):
        return False
    import requests
    endpoint = f"{url.rstrip('/')}/rest/v1/{table}"
    params = {"on_conflict": on_conflict} if on_conflict else {}
    headers = {
        "apikey": key, "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    ok = True
    for i in range(0, len(rows), 500):
        r = requests.post(endpoint, params=params, headers=headers,
                          data=json.dumps(rows[i:i + 500]), timeout=30)
        if r.status_code >= 400:
            print(f"  supabase {table}: {r.status_code} {r.text[:200]}", file=sys.stderr)
            ok = False
    return ok


def main() -> int:
    print(f"[crawl_news] {dt.datetime.utcnow().isoformat()}Z")

    # 1. fresh live data ------------------------------------------------------
    try:
        from scripts.fetch_live import fetch_live, DEFAULT_OUT  # type: ignore
    except ImportError:
        from fetch_live import fetch_live, DEFAULT_OUT  # run from scripts/
    try:
        fetch_live(os.path.abspath(DEFAULT_OUT))
    except Exception as e:
        print(f"  live fetch failed ({e}); using existing files", file=sys.stderr)

    from recommender import live_fpl, news_aggregator
    try:
        live = live_fpl.LiveData.load()
    except FileNotFoundError:
        print("  no live data available at all; aborting gracefully")
        return 0

    hrs = hours_to_deadline(live)
    print(f"  hours to next kickoff: {hrs if hrs is not None else 'unknown'}")
    urgent = hrs is not None and 0 <= hrs <= DEADLINE_WINDOW_HOURS

    # 2. gather news -----------------------------------------------------------
    pool = live_fpl.build_pool()
    try:
        by_name = news_aggregator.gather(pool, use_rss=True)
    except Exception as e:
        print(f"  news gather failed: {e}", file=sys.stderr)
        by_name = {}
    flat = news_aggregator.flatten(by_name)
    print(f"  news items: {len(flat)} ({'DEADLINE WINDOW' if urgent else 'routine run'})")

    gw = live.current_gameweek()

    # 3. persist ---------------------------------------------------------------
    #    Supabase (shared, for the deployed app) if configured...
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    news_rows = [{
        "fetched_at": now, "season": None, "gameweek": gw,
        "player": n.get("player"), "source": n.get("source"),
        "reliability": n.get("reliability"), "corroboration": n.get("corroboration"),
        "category": n.get("category"), "recency": n.get("recency"),
        "headline": n.get("headline"), "detail": n.get("detail"),
    } for n in flat]
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    teams = {t["id"]: t for t in live.bootstrap["teams"]}
    player_rows = []
    for e in live.bootstrap["elements"]:
        t = teams.get(e["team"], {})
        def f(k):
            try: return float(e.get(k))
            except (TypeError, ValueError): return None
        player_rows.append({
            "player_id": e["id"], "fetched_at": now,
            "name": f'{e.get("first_name","")} {e.get("second_name","")}'.strip(),
            "web_name": e.get("web_name"), "team": t.get("name"),
            "team_code": t.get("code"), "position": pos_map.get(e["element_type"], "MID"),
            "price": e["now_cost"] / 10.0, "form": f("form"), "ppg": f("points_per_game"),
            "ep_next": f("ep_next"), "status": e.get("status"),
            "chance": e.get("chance_of_playing_next_round"), "news": e.get("news"),
        })
    fixture_rows = [{
        "fixture_id": fx["id"], "fetched_at": now, "event": fx.get("event"),
        "team_h": fx["team_h"], "team_a": fx["team_a"],
        "team_h_difficulty": fx.get("team_h_difficulty"),
        "team_a_difficulty": fx.get("team_a_difficulty"),
        "finished": bool(fx.get("finished")), "kickoff": fx.get("kickoff_time"),
    } for fx in live.fixtures]

    if os.getenv("SUPABASE_URL"):
        print("  upserting to Supabase...")
        upsert_supabase("live_players", player_rows, on_conflict="player_id")
        upsert_supabase("live_fixtures", fixture_rows, on_conflict="fixture_id")
        upsert_supabase("news", news_rows)
    else:
        print("  SUPABASE_URL not set - persisting locally only")

    #    ...and always locally (SQLite + files) for a self-hosted instance.
    try:
        import database as db
        db.ingest_live()
        if flat:
            db.save_news(flat, None, gw)
        print("  local SQLite updated")
    except Exception as e:
        print(f"  local db skip: {e}", file=sys.stderr)

    print("[crawl_news] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
