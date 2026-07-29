"""
Build and populate the SQLite database from existing data.

Run locally after you've fetched/processed seasons and fetched live data:
    python3 scripts/init_db.py

Idempotent - safe to re-run any time to refresh the DB from the CSV/JSON files.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import database as db


def main():
    db.init_schema()
    print(f"Database: {db.DB_PATH}")

    print("\nIngesting processed seasons...")
    counts = db.ingest_all_processed()
    for season, n in counts.items():
        print(f"  {season}: {n:,} player-gameweek rows")

    print("\nIngesting live snapshot...")
    live = db.ingest_live()
    print(f"  live_players: {live['players']:,} | live_fixtures: {live['fixtures']:,}")

    print("\nDatabase contents:")
    for k, v in db.stats().items():
        print(f"  {k}: {v}")
    print("\nDone.")


if __name__ == "__main__":
    main()
