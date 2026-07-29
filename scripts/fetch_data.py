"""
Fetch historical Fantasy Premier League data for backtesting.

Data source: vaastav/Fantasy-Premier-League (public GitHub archive of every
past FPL season, gameweek-by-gameweek player stats).
https://github.com/vaastav/Fantasy-Premier-League

Run locally (needs normal internet access):
    python3 fetch_data.py --season 2024-25
    python3 fetch_data.py --seasons 2020-21,2021-22,2022-23,2023-24,2024-25
    python3 fetch_data.py --recent 5   # shortcut for the last 5 completed seasons

Saves into ../data/raw/<season>/:
    merged_gw.csv   - every player, every gameweek, that season (main dataset)
    teams.csv       - team id -> name mapping
    fixtures.csv    - fixture list with difficulty ratings
    players_raw.csv - one row per player, season-level summary (bootstrap-static snapshot)
"""

import argparse
import os
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

FILES = {
    "merged_gw.csv": "gws/merged_gw.csv",
    "teams.csv": "teams.csv",
    "fixtures.csv": "fixtures.csv",
    "players_raw.csv": "players_raw.csv",
}

# Ordered list of completed-season folders available in the repo, oldest to
# newest, used by --recent N. Update this if a newer completed season isn't
# listed yet - the repo is community-maintained and can lag a bit after a
# season actually ends.
KNOWN_SEASONS = [
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21",
    "2021-22", "2022-23", "2023-24", "2024-25",
]


def download(url: str, dest_path: str) -> None:
    print(f"  {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "fpl-agent/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    with open(dest_path, "wb") as f:
        f.write(data)
    print(f"    -> saved {len(data):,} bytes to {dest_path}")


def fetch_season(season: str, out_dir: str) -> None:
    season_dir = os.path.join(out_dir, season)
    os.makedirs(season_dir, exist_ok=True)

    print(f"Fetching {season} season data into {season_dir}/")
    for local_name, remote_path in FILES.items():
        url = f"{BASE_URL}/{season}/{remote_path}"
        dest = os.path.join(season_dir, local_name)
        try:
            download(url, dest)
        except Exception as e:
            print(f"    WARNING: failed to fetch {remote_path}: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--season",
        help="Single season folder name as used in the repo, e.g. 2024-25",
    )
    group.add_argument(
        "--seasons",
        help="Comma-separated list of seasons, e.g. 2022-23,2023-24,2024-25",
    )
    group.add_argument(
        "--recent",
        type=int,
        help="Fetch the N most recently completed seasons (e.g. --recent 5)",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "..", "data", "raw"),
        help="Where to save downloaded files (default: ../data/raw)",
    )
    args = parser.parse_args()

    if args.seasons:
        seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    elif args.recent:
        seasons = KNOWN_SEASONS[-args.recent:]
    else:
        seasons = [args.season or "2024-25"]

    out_dir = os.path.abspath(args.out_dir)
    for i, season in enumerate(seasons, 1):
        print(f"[{i}/{len(seasons)}]")
        fetch_season(season, out_dir)
        print()
    print("Done.")


if __name__ == "__main__":
    main()
