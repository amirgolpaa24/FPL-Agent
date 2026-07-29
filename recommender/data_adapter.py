"""
Turn the processed feature table into PlayerFeatures objects for a given
gameweek. Used by tests and the Streamlit page to get realistic inputs; a
live deployment would instead build PlayerFeatures from the current FPL API.
"""

from __future__ import annotations

import os

import pandas as pd

from .features import PlayerFeatures, Squad

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)


def _features_path(season: str) -> str:
    return os.path.join(_DATA_DIR, season, "player_gw_features.csv")


def load_gameweek(season: str, gameweek: int) -> pd.DataFrame:
    # Prefer the SQLite database when it holds this season; fall back to the
    # processed CSV otherwise. Same shape either way (a `GW` column).
    try:
        import database as db
        if db.has_season(season):
            gw = db.get_gameweek(season, gameweek)
            if not gw.empty:
                return gw
    except Exception:
        pass
    df = pd.read_csv(_features_path(season))
    return df[df["GW"] == gameweek].copy()


def _row_to_player(row: pd.Series) -> PlayerFeatures:
    def g(col):
        v = row.get(col)
        return None if pd.isna(v) else v
    return PlayerFeatures(
        name=row["name"],
        team=row["team"],
        position=row["position"],
        price=float(row["value"]) / 10.0,   # feature table stores price x10
        form_points=g("form_points"),
        season_form_points=g("season_form_points"),
        last_gw_points=g("last_gw_points"),
        form_ict=g("form_ict"),
        form_minutes=g("form_minutes"),
        fixture_difficulty=g("fixture_difficulty"),
        opp_goals_conceded_form=g("opp_goals_conceded_form"),
        was_home=bool(row["was_home"]) if not pd.isna(row.get("was_home")) else None,
        gameweek=int(row["GW"]),
    )


def player_pool(season: str, gameweek: int, min_minutes: float = 0.0) -> list[PlayerFeatures]:
    """All players available in a given gameweek as PlayerFeatures."""
    gw = load_gameweek(season, gameweek)
    if min_minutes:
        gw = gw[gw["form_minutes"].fillna(0) >= min_minutes]
    return [_row_to_player(r) for _, r in gw.iterrows()]


def sample_squad(season: str, gameweek: int, bank: float = 0.0,
                 free_transfers: int = 1, budget: float = 100.0) -> Squad:
    """Build a plausible legal 15-man squad from a gameweek, for demos/tests:
    greedily takes the best season-form players per position while always
    reserving enough budget to fill the remaining slots with the cheapest
    legal players, so the total never exceeds `budget`. Respects the
    max-3-per-club rule."""
    pool = player_pool(season, gameweek, min_minutes=1)
    ranked = sorted(pool, key=lambda p: (p.season_form_points or 0), reverse=True)

    quota = dict(Squad.QUOTA)
    by_club: dict[str, int] = {}
    chosen: list[PlayerFeatures] = []
    chosen_names: set[str] = set()
    spent = 0.0

    def min_cost_to_fill(remaining: dict, exclude: set[str]) -> float:
        total = 0.0
        for pos, need in remaining.items():
            if need <= 0:
                continue
            prices = sorted(
                p.price for p in pool
                if p.position == pos and p.name not in exclude
            )
            if len(prices) < need:
                return float("inf")
            total += sum(prices[:need])
        return total

    for p in ranked:
        if quota.get(p.position, 0) <= 0 or p.name in chosen_names:
            continue
        if by_club.get(p.team, 0) >= Squad.MAX_PER_CLUB:
            continue
        remaining = {pos: (n - 1 if pos == p.position else n) for pos, n in quota.items()}
        if spent + p.price + min_cost_to_fill(remaining, chosen_names | {p.name}) > budget + 1e-6:
            continue
        chosen.append(p)
        chosen_names.add(p.name)
        spent += p.price
        quota[p.position] -= 1
        by_club[p.team] = by_club.get(p.team, 0) + 1
        if all(v == 0 for v in quota.values()):
            break

    return Squad(players=chosen, bank=bank, free_transfers=free_transfers)
