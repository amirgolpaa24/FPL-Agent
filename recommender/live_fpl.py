"""
Live current-season adapter: turn the official FPL API payloads (fetched by
scripts/fetch_live.py) into PlayerFeatures the engine can score, with:

  - next-gameweek fixture difficulty per player,
  - within-season form straight from the API (`form`, `points_per_game`),
  - cold-start seeding from a prior season for the opening weeks,
  - AUTO-NEWS: the API's own injury/availability fields (`status`, `news`,
    `chance_of_playing_next_round`) are turned into NewsItems, so even before
    you type any news yourself the LLM mode sees official availability flags.

This is the bridge from "backtesting on history" to "using it live".
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .features import PlayerFeatures, NewsItem, Squad
from . import coldstart

_LIVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "live"
)

# FPL element_type ids -> our position codes
POS_MAP = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# status codes -> (category, human phrase)
STATUS_MAP = {
    "i": ("injury", "injured"),
    "s": ("suspension", "suspended"),
    "d": ("injury", "doubtful / fitness concern"),
    "u": ("availability", "unavailable"),
    "n": ("availability", "not in squad / ineligible"),
}


def _load(name: str) -> dict:
    path = os.path.join(_LIVE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run scripts/fetch_live.py first to download live data."
        )
    with open(path) as f:
        return json.load(f)


class LiveData:
    """Parsed handle over the fetched FPL payloads."""

    def __init__(self, bootstrap: dict, fixtures: list):
        self.bootstrap = bootstrap
        self.fixtures = fixtures
        self.teams = {t["id"]: t for t in bootstrap["teams"]}
        self.events = bootstrap.get("events", [])

    @classmethod
    def load(cls) -> "LiveData":
        return cls(_load("bootstrap.json"), _load("fixtures.json"))

    # -- gameweeks ---------------------------------------------------------
    def current_gameweek(self) -> int:
        cur = next((e["id"] for e in self.events if e.get("is_current")), None)
        nxt = next((e["id"] for e in self.events if e.get("is_next")), None)
        return cur or nxt or 1

    def season_started(self) -> bool:
        return any(e.get("finished") for e in self.events)

    def freshness(self) -> dict:
        """Detect whether the API has rolled over to a new (upcoming) season
        yet, or is still serving a completed one. During the off-season the
        public API keeps returning the finished season until the new fixture
        list is published, so a 'GW1' decision would silently use OLD data."""
        n = len(self.events)
        finished = sum(1 for e in self.events if e.get("finished"))
        has_next = any(e.get("is_next") for e in self.events)
        # any fixture not yet finished => a real upcoming schedule exists
        upcoming_fixtures = sum(1 for f in self.fixtures if not f.get("finished"))
        stale = (n > 0 and finished == n and not has_next and upcoming_fixtures == 0)
        return {
            "stale": stale,
            "events_total": n,
            "events_finished": finished,
            "has_next_gw": has_next,
            "upcoming_fixtures": upcoming_fixtures,
        }

    # -- fixtures ----------------------------------------------------------
    def team_fixture_in_gw(self, team_id: int, gw: int) -> Optional[dict]:
        """Return (difficulty, is_home) for a team's fixture in a gameweek,
        or None if they don't play (blank) or play twice (returns the first)."""
        for fx in self.fixtures:
            if fx.get("event") != gw:
                continue
            if fx["team_h"] == team_id:
                return {"difficulty": fx["team_h_difficulty"], "is_home": True,
                        "opponent": fx["team_a"]}
            if fx["team_a"] == team_id:
                return {"difficulty": fx["team_a_difficulty"], "is_home": False,
                        "opponent": fx["team_h"]}
        return None


def _auto_news(el: dict) -> list[NewsItem]:
    """Build NewsItems from the API's own availability fields."""
    items = []
    status = el.get("status", "a")
    if status != "a" and status in STATUS_MAP:
        cat, phrase = STATUS_MAP[status]
        chance = el.get("chance_of_playing_next_round")
        detail = el.get("news", "") or ""
        if chance is not None:
            detail = f"{detail} (chance of playing: {chance}%)".strip()
        # negative sentiment scaled by how unlikely they are to play
        sentiment = -1.0 if chance in (0, None) and status in ("i", "s", "u") else -0.5
        items.append(NewsItem(
            headline=f"Official status: {phrase}",
            detail=detail, source="FPL API", category=cat, sentiment=sentiment,
        ))
    elif el.get("news"):
        # sometimes there's news even when nominally available (e.g. "knock")
        items.append(NewsItem(
            headline=el["news"], source="FPL API", category="general",
        ))
    return items


def _element_to_player(el: dict, live: LiveData, gw: int) -> PlayerFeatures:
    team = live.teams[el["team"]]
    full_name = f"{el.get('first_name', '')} {el.get('second_name', '')}".strip()
    fixture = live.team_fixture_in_gw(el["team"], gw)

    def fnum(key):
        v = el.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    ppg = fnum("points_per_game")
    form = fnum("form")
    minutes = fnum("minutes")
    # approximate recent minutes-per-game from season minutes / games played
    starts = el.get("starts") or 0
    games = max(starts, 1)
    form_minutes = (minutes / games) if minutes else None

    # Availability from the API: chance_of_playing_next_round is 0-100 (or null
    # when fully fit). status 'a' = available. Injured/suspended/unavailable
    # with no chance given -> treat as 0.
    chance = el.get("chance_of_playing_next_round")
    status = el.get("status", "a")
    if chance is not None:
        availability = chance / 100.0
    elif status == "a":
        availability = 1.0
    else:
        availability = 0.0

    ep_next = fnum("ep_next")  # FPL's own official expected points, next GW

    return PlayerFeatures(
        name=full_name or el.get("web_name", "unknown"),
        team=team["name"],
        position=POS_MAP.get(el["element_type"], "MID"),
        price=el["now_cost"] / 10.0,
        form_points=form,
        season_form_points=ppg,
        last_gw_points=fnum("event_points"),
        form_ict=fnum("ict_index"),
        form_minutes=form_minutes,
        fixture_difficulty=(fixture["difficulty"] if fixture else None),
        was_home=(fixture["is_home"] if fixture else None),
        availability=availability,
        fpl_ep_next=ep_next,
        player_id=el["id"],
        team_code=team.get("code"),
        gameweek=gw,
        news=_auto_news(el),
    )


def build_pool(
    gameweek: Optional[int] = None,
    prior_season: Optional[str] = None,
    only_available: bool = False,
) -> list[PlayerFeatures]:
    """Build the full current-season player pool as PlayerFeatures.

    gameweek:      target GW (defaults to the current/next one).
    prior_season:  processed season name to cold-start from for early GWs
                   (e.g. "2025-26"). Strongly recommended for GW1-5.
    only_available: drop players flagged unavailable/injured by the API.
    """
    live = LiveData.load()
    gw = gameweek or live.current_gameweek()

    elements = live.bootstrap["elements"]
    if only_available:
        elements = [e for e in elements if e.get("status") == "a"]

    players = [_element_to_player(e, live, gw) for e in elements]

    if prior_season and coldstart.blended_weight(gw) > 0:
        lookup = coldstart._lookup(coldstart.prior_season_table(prior_season))
        players = [coldstart.seed_player(p, lookup, gw) for p in players]

    return players


ENTRY_TTL_SECONDS = 15 * 60  # re-fetch a user's team at most every 15 minutes
_FPL_API = "https://fantasy.premierleague.com/api"


def fetch_entry(entry_id: int, force: bool = False) -> dict:
    """Fetch a user's FPL entry + current-GW picks straight from the FPL API,
    with a short on-disk cache. This is what lets a regular user just type
    their team ID in the app - no scripts, the SERVER does the fetching.

    Raises LookupError if the team id doesn't exist. Missing picks (FPL only
    publishes them after the gameweek deadline) are NOT an error here - the
    caller checks for that and explains it."""
    import time
    import urllib.error
    import urllib.request

    path = os.path.join(_LIVE_DIR, f"entry_{entry_id}.json")
    if not force and os.path.exists(path) and \
            time.time() - os.path.getmtime(path) < ENTRY_TTL_SECONDS:
        with open(path) as f:
            return json.load(f)

    def _get(url: str) -> dict:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fpl-agent)"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read())

    live = LiveData.load()
    gw = live.current_gameweek()
    try:
        entry = _get(f"{_FPL_API}/entry/{entry_id}/")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise LookupError(f"FPL team {entry_id} doesn't exist") from e
        raise
    try:
        picks = _get(f"{_FPL_API}/entry/{entry_id}/event/{gw}/picks/")
    except Exception:
        picks = {}  # pre-deadline: FPL hasn't published picks yet

    data = {"entry": entry, "picks": picks, "gameweek": gw}
    try:
        os.makedirs(_LIVE_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
    except OSError:
        pass  # caching is best-effort
    return data


def build_squad_from_entry(
    entry_id: int,
    gameweek: Optional[int] = None,
    prior_season: Optional[str] = None,
) -> Squad:
    """Reconstruct YOUR 15-man squad + bank for a team id. Fetches live from
    the FPL API (cached); falls back to a previously saved file if the network
    call fails."""
    try:
        data = fetch_entry(int(entry_id))
    except LookupError:
        raise
    except Exception:
        data = _load(f"entry_{entry_id}.json")  # offline fallback if we have one
    live = LiveData.load()
    gw = gameweek or data.get("gameweek") or live.current_gameweek()

    by_id = {e["id"]: e for e in live.bootstrap["elements"]}
    picks = (data.get("picks") or {}).get("picks", [])
    if not picks:
        raise ValueError(
            f"FPL hasn't published this team's GW{gw} picks yet - they only become "
            "available after the gameweek deadline."
        )

    players = []
    for pick in picks:
        el = by_id.get(pick["element"])
        if el:
            players.append(_element_to_player(el, live, gw))

    if prior_season and coldstart.blended_weight(gw) > 0:
        lookup = coldstart._lookup(coldstart.prior_season_table(prior_season))
        players = [coldstart.seed_player(p, lookup, gw) for p in players]

    history = (data.get("picks") or {}).get("entry_history", {})
    bank = history.get("bank", 0) / 10.0  # bank is in tenths of a million

    return Squad(players=players, bank=bank, free_transfers=1)
