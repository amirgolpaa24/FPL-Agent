"""
Cold-start seeding for the start of a season.

At gameweek 1 there is no within-season form yet - every rolling feature is
empty - so the engine has nothing to score on. The fix is to seed those
features from the PRIOR season's per-game numbers, matched by player name,
then fade the prior-season signal out as real gameweeks accumulate.

This mirrors what the Squad Builder analysis page does for squad selection,
but packaged so the live recommender can use it for the opening weeks of a
real season.

Design:
  - prior_season_table(): a name -> (points_per_game, avg_minutes) map from a
    processed prior season.
  - seed_player(): fill a PlayerFeatures' missing form fields from that map.
  - blended_weight(): how much to trust prior-season vs within-season form at
    a given gameweek (prior dominates at GW1, gone by ~GW6).
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import Optional

import pandas as pd

from .features import PlayerFeatures

_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)

# By this gameweek, within-season form fully replaces the prior-season prior.
BLEND_FADE_BY_GW = 6


def prior_season_table(prior_season: str) -> pd.DataFrame:
    """Per-player summary of a completed prior season: points-per-game and
    average minutes among gameweeks they featured in. Keyed by full name,
    which matches the naming used across the processed feature tables."""
    path = os.path.join(_DATA_DIR, prior_season, "player_gw_features.csv")
    df = pd.read_csv(path)

    played = df[df["minutes"] > 0]
    grp = played.groupby("name")
    table = pd.DataFrame({
        "prior_ppg": grp["total_points"].mean(),
        "prior_avg_minutes": grp["minutes"].mean(),
        "prior_games": grp["total_points"].count(),
    })
    # season total across ALL rows, reindexed onto the played-players index so
    # columns stay aligned (players who never played are dropped entirely).
    table["prior_total"] = df.groupby("name")["total_points"].sum().reindex(table.index)
    table = table.reset_index()
    # Only keep players with a real per-game record and enough sample to trust.
    table = table.dropna(subset=["prior_ppg", "prior_avg_minutes"])
    return table


def _lookup(table: pd.DataFrame) -> dict:
    return {
        r["name"].strip().lower(): r
        for _, r in table.iterrows()
    }


def blended_weight(gameweek: int, fade_by: int = BLEND_FADE_BY_GW) -> float:
    """Weight on the PRIOR-season signal at a given gameweek: 1.0 at GW1,
    linearly to 0.0 at `fade_by`. (Within-season weight is 1 - this.)"""
    if gameweek <= 1:
        return 1.0
    if gameweek >= fade_by:
        return 0.0
    return (fade_by - gameweek) / (fade_by - 1)


def seed_player(
    player: PlayerFeatures,
    prior_lookup: dict,
    gameweek: int,
    fade_by: int = BLEND_FADE_BY_GW,
) -> PlayerFeatures:
    """Return a copy of `player` with form fields seeded/blended from the
    prior season. If the player has no prior-season record (new signing,
    promoted-club player), their within-season values are left as-is - which
    at GW1 means they'll score low, correctly reflecting genuine uncertainty."""
    w_prior = blended_weight(gameweek, fade_by)
    if w_prior <= 0:
        return player

    rec = prior_lookup.get(player.name.strip().lower())
    if rec is None:
        return player

    prior_ppg = float(rec["prior_ppg"])
    prior_min = float(rec["prior_avg_minutes"])
    if pd.isna(prior_ppg) or pd.isna(prior_min):
        return player

    def blend(current: Optional[float], prior: float) -> float:
        cur = 0.0 if current is None or pd.isna(current) else float(current)
        return w_prior * prior + (1 - w_prior) * cur

    return replace(
        player,
        season_form_points=blend(player.season_form_points, prior_ppg),
        form_points=blend(player.form_points, prior_ppg),
        last_gw_points=(player.last_gw_points if player.last_gw_points is not None else prior_ppg),
        form_minutes=blend(player.form_minutes, prior_min),
    )


def seed_pool(
    players: list[PlayerFeatures],
    prior_season: str,
    gameweek: int,
    fade_by: int = BLEND_FADE_BY_GW,
) -> list[PlayerFeatures]:
    """Seed an entire player pool for the given gameweek."""
    if blended_weight(gameweek, fade_by) <= 0:
        return players
    lookup = _lookup(prior_season_table(prior_season))
    return [seed_player(p, lookup, gameweek, fade_by) for p in players]
