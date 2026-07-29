"""
SQLite storage layer for FPL Agent.

Why SQLite: the data is relational (players belong to teams; stats belong to a
player+gameweek; news belongs to a player+time+source; recommendations
reference a gameweek), and the app is a local, single-user desktop tool.
SQLite is a serverless, single-file relational database that ships with Python
- zero setup, full SQL (joins/aggregations/indexes), and easily migrated to
Postgres later (same SQL/ORM) if this ever becomes a multi-user service.

Everything here uses only the stdlib `sqlite3` (+ pandas, already a dependency),
so there's nothing extra to install. The database file lives at
data/fpl_agent.db and is created on first use.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(ROOT, "data", "fpl_agent.db")
_PROCESSED = os.path.join(ROOT, "data", "processed")
_LIVE = os.path.join(ROOT, "data", "live")

SCHEMA = """
CREATE TABLE IF NOT EXISTS player_gw (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season TEXT, gameweek INTEGER, name TEXT, position TEXT, team TEXT,
    opponent_team INTEGER, was_home INTEGER, value REAL, minutes INTEGER,
    total_points INTEGER, form_points REAL, form_minutes REAL, form_ict REAL,
    last_gw_points REAL, season_form_points REAL, opp_goals_conceded_form REAL,
    opp_goals_scored_form REAL, fixture_difficulty REAL
    -- (season, gameweek, name) is NOT unique: double-gameweeks give a player
    -- two rows in one GW. A surrogate key handles that; queries use the index.
);
CREATE INDEX IF NOT EXISTS idx_pgw_season_gw ON player_gw(season, gameweek);
CREATE INDEX IF NOT EXISTS idx_pgw_name ON player_gw(name);

CREATE TABLE IF NOT EXISTS teams (
    season TEXT, id INTEGER, code INTEGER, name TEXT, short_name TEXT,
    PRIMARY KEY (season, id)
);

CREATE TABLE IF NOT EXISTS live_players (
    fetched_at TEXT, player_id INTEGER, name TEXT, web_name TEXT, team TEXT,
    team_code INTEGER, position TEXT, price REAL, form REAL, ppg REAL,
    ep_next REAL, status TEXT, chance INTEGER, news TEXT,
    PRIMARY KEY (player_id)
);

CREATE TABLE IF NOT EXISTS live_fixtures (
    fetched_at TEXT, fixture_id INTEGER, event INTEGER, team_h INTEGER, team_a INTEGER,
    team_h_difficulty INTEGER, team_a_difficulty INTEGER, finished INTEGER, kickoff TEXT,
    PRIMARY KEY (fixture_id)
);

CREATE TABLE IF NOT EXISTS news (
    id INTEGER PRIMARY KEY AUTOINCREMENT, fetched_at TEXT, season TEXT, gameweek INTEGER,
    player TEXT, source TEXT, reliability REAL, corroboration INTEGER,
    category TEXT, recency TEXT, headline TEXT, detail TEXT
);
CREATE INDEX IF NOT EXISTS idx_news_player ON news(player);

CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, decision TEXT, mode TEXT,
    source TEXT, season TEXT, gameweek INTEGER, formula TEXT,
    request_json TEXT, result_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_rec_decision ON recommendations(decision, created_at);
"""

PGW_COLUMNS = [
    "name", "position", "team", "opponent_team", "was_home", "value", "minutes",
    "total_points", "form_points", "form_minutes", "form_ict", "last_gw_points",
    "season_form_points", "opp_goals_conceded_form", "opp_goals_scored_form",
    "fixture_difficulty",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_schema(conn: sqlite3.Connection | None = None) -> None:
    own = conn is None
    conn = conn or connect()
    conn.executescript(SCHEMA)
    conn.commit()
    if own:
        conn.close()


def db_exists() -> bool:
    return os.path.exists(DB_PATH)


# ---------------------------------------------------------------------------
# ingestion
# ---------------------------------------------------------------------------

def ingest_processed_season(season: str, conn: sqlite3.Connection | None = None) -> int:
    """Load one processed season's feature CSV into player_gw (idempotent)."""
    path = os.path.join(_PROCESSED, season, "player_gw_features.csv")
    if not os.path.exists(path):
        return 0
    own = conn is None
    conn = conn or connect()
    init_schema(conn)

    df = pd.read_csv(path)
    df = df.rename(columns={"GW": "gameweek"})
    df["season"] = season
    if df["was_home"].dtype == bool:
        df["was_home"] = df["was_home"].astype(int)
    keep = ["season", "gameweek"] + [c for c in PGW_COLUMNS if c in df.columns]
    df = df[keep]

    conn.execute("DELETE FROM player_gw WHERE season=?", (season,))
    df.to_sql("player_gw", conn, if_exists="append", index=False)
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM player_gw WHERE season=?", (season,)).fetchone()[0]
    if own:
        conn.close()
    return n


def ingest_all_processed() -> dict[str, int]:
    if not os.path.isdir(_PROCESSED):
        return {}
    conn = connect()
    init_schema(conn)
    out = {}
    for season in sorted(os.listdir(_PROCESSED)):
        if os.path.exists(os.path.join(_PROCESSED, season, "player_gw_features.csv")):
            out[season] = ingest_processed_season(season, conn)
    conn.close()
    return out


def ingest_live(conn: sqlite3.Connection | None = None) -> dict:
    """Snapshot the current live bootstrap + fixtures into the DB."""
    bpath = os.path.join(_LIVE, "bootstrap.json")
    fpath = os.path.join(_LIVE, "fixtures.json")
    if not os.path.exists(bpath):
        return {"players": 0, "fixtures": 0}
    own = conn is None
    conn = conn or connect()
    init_schema(conn)
    ts = _now()

    boot = json.load(open(bpath))
    teams = {t["id"]: t for t in boot["teams"]}
    pos = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    conn.execute("DELETE FROM live_players")
    for e in boot["elements"]:
        t = teams.get(e["team"], {})
        conn.execute(
            "INSERT OR REPLACE INTO live_players VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, e["id"], f"{e.get('first_name','')} {e.get('second_name','')}".strip(),
             e.get("web_name"), t.get("name"), t.get("code"),
             pos.get(e["element_type"], "MID"), e["now_cost"] / 10.0,
             _f(e.get("form")), _f(e.get("points_per_game")), _f(e.get("ep_next")),
             e.get("status"), e.get("chance_of_playing_next_round"), e.get("news")),
        )

    # teams (tag with a 'live' season marker)
    for t in boot["teams"]:
        conn.execute("INSERT OR REPLACE INTO teams VALUES (?,?,?,?,?)",
                     ("live", t["id"], t["code"], t["name"], t.get("short_name")))

    nfix = 0
    if os.path.exists(fpath):
        conn.execute("DELETE FROM live_fixtures")
        for fx in json.load(open(fpath)):
            conn.execute(
                "INSERT OR REPLACE INTO live_fixtures VALUES (?,?,?,?,?,?,?,?,?)",
                (ts, fx["id"], fx.get("event"), fx["team_h"], fx["team_a"],
                 fx.get("team_h_difficulty"), fx.get("team_a_difficulty"),
                 int(bool(fx.get("finished"))), fx.get("kickoff_time")),
            )
            nfix += 1
    conn.commit()
    nplayers = conn.execute("SELECT COUNT(*) FROM live_players").fetchone()[0]
    if own:
        conn.close()
    return {"players": nplayers, "fixtures": nfix, "fetched_at": ts}


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# writes: news + recommendation history
# ---------------------------------------------------------------------------

def save_news(news_used: list[dict], season: str | None, gameweek: int | None) -> int:
    if not news_used:
        return 0
    conn = connect()
    init_schema(conn)
    ts = _now()
    for n in news_used:
        conn.execute(
            "INSERT INTO news (fetched_at, season, gameweek, player, source, reliability,"
            " corroboration, category, recency, headline, detail) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (ts, season, gameweek, n.get("player"), n.get("source"), n.get("reliability"),
             n.get("corroboration"), n.get("category"), n.get("recency"),
             n.get("headline"), n.get("detail")),
        )
    conn.commit()
    conn.close()
    return len(news_used)


def log_recommendation(request: dict, result: dict) -> int:
    conn = connect()
    init_schema(conn)
    meta = result.get("_meta", {})
    # keep the stored result compact (drop bulky nested anchors)
    slim = {k: v for k, v in result.items() if k != "stats_anchor"}
    cur = conn.execute(
        "INSERT INTO recommendations (created_at, decision, mode, source, season,"
        " gameweek, formula, request_json, result_json) VALUES (?,?,?,?,?,?,?,?,?)",
        (_now(), meta.get("decision"), meta.get("mode"), meta.get("source"),
         meta.get("season"), meta.get("gameweek"), meta.get("formula"),
         json.dumps(request)[:20000], json.dumps(slim, default=str)[:40000]),
    )
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid


# ---------------------------------------------------------------------------
# reads
# ---------------------------------------------------------------------------

def has_season(season: str) -> bool:
    if not db_exists():
        return False
    conn = connect()
    try:
        row = conn.execute("SELECT 1 FROM player_gw WHERE season=? LIMIT 1", (season,)).fetchone()
        return row is not None
    finally:
        conn.close()


def list_seasons() -> list[str]:
    if not db_exists():
        return []
    conn = connect()
    try:
        rows = conn.execute("SELECT DISTINCT season FROM player_gw ORDER BY season").fetchall()
        return [r[0] for r in rows]
    finally:
        conn.close()


def get_gameweek(season: str, gameweek: int) -> pd.DataFrame:
    """Return one gameweek's rows as a DataFrame shaped like the CSV
    (a `GW` column), so it's a drop-in for the file-based loader."""
    conn = connect()
    try:
        df = pd.read_sql_query(
            "SELECT *, gameweek AS GW FROM player_gw WHERE season=? AND gameweek=?",
            conn, params=(season, gameweek),
        )
    finally:
        conn.close()
    if "was_home" in df.columns:
        df["was_home"] = df["was_home"].astype(bool)
    return df


def recent_recommendations(limit: int = 20) -> pd.DataFrame:
    if not db_exists():
        return pd.DataFrame()
    conn = connect()
    try:
        return pd.read_sql_query(
            "SELECT id, created_at, decision, mode, source, season, gameweek, formula"
            " FROM recommendations ORDER BY id DESC LIMIT ?", conn, params=(limit,))
    finally:
        conn.close()


def stats() -> dict:
    """Row counts per table, for a health/overview view."""
    if not db_exists():
        return {"exists": False}
    conn = connect()
    out = {"exists": True, "path": DB_PATH}
    try:
        for tbl in ["player_gw", "teams", "live_players", "live_fixtures", "news", "recommendations"]:
            try:
                out[tbl] = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
            except sqlite3.OperationalError:
                out[tbl] = 0
    finally:
        conn.close()
    return out
