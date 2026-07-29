"""
Fetch LIVE current-season data from the official Fantasy Premier League API.

Unlike fetch_data.py (which pulls the historical community archive), this hits
the real FPL endpoints for the season currently in progress - current prices,
the current player list, live fixtures, injury/availability status, and
optionally YOUR squad by entry id.

Run locally (needs internet; the FPL API is public, no key required):
    python3 fetch_live.py
    python3 fetch_live.py --entry 1234567     # also pull your team

Saves into ../data/live/:
    bootstrap.json   - players, teams, positions, gameweeks (the main payload)
    fixtures.json    - all fixtures with official difficulty ratings
    entry_<id>.json  - your squad + bank + picks (only with --entry)

The recommender/live_fpl.py adapter reads these files. Re-run this whenever you
want fresh prices/news (e.g. before making a gameweek decision).
"""

import argparse
import json
import os
import urllib.request

BASE = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0 (fpl-agent)"}

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "..", "data", "live")


def _get(url: str) -> dict:
    print(f"  GET {url}")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read())


def _save(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"    -> {path} ({os.path.getsize(path):,} bytes)")


def fetch_live(out_dir: str, entry_id: int | None = None) -> None:
    os.makedirs(out_dir, exist_ok=True)

    bootstrap = _get(f"{BASE}/bootstrap-static/")
    _save(bootstrap, os.path.join(out_dir, "bootstrap.json"))

    fixtures = _get(f"{BASE}/fixtures/")
    _save(fixtures, os.path.join(out_dir, "fixtures.json"))

    if entry_id:
        # Determine the current/next gameweek to pull the right picks.
        events = bootstrap.get("events", [])
        cur = next((e["id"] for e in events if e.get("is_current")), None)
        nxt = next((e["id"] for e in events if e.get("is_next")), None)
        gw = cur or nxt or 1

        entry = _get(f"{BASE}/entry/{entry_id}/")
        try:
            picks = _get(f"{BASE}/entry/{entry_id}/event/{gw}/picks/")
        except Exception as e:
            print(f"    (no picks for GW{gw} yet: {e})")
            picks = {}
        _save({"entry": entry, "picks": picks, "gameweek": gw},
              os.path.join(out_dir, f"entry_{entry_id}.json"))

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry", type=int, default=None,
                        help="Your FPL entry (team) id - found in your team's URL")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    args = parser.parse_args()
    fetch_live(os.path.abspath(args.out_dir), args.entry)


if __name__ == "__main__":
    main()
