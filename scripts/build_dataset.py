"""
Turn raw merged_gw.csv into a clean, lookahead-safe per-player-per-gameweek
feature table for backtesting.

CRITICAL RULE: every feature for gameweek N must only use information that
was actually known before gameweek N kicked off. All rolling stats are
computed on a *shifted* series (i.e. they exclude the current row) so the
backtest can't accidentally "predict" using the outcome it's trying to predict.

Usage:
    python3 build_dataset.py --season 2024-25 --window 4

Reads:  ../data/raw/<season>/merged_gw.csv
Writes: ../data/processed/<season>/player_gw_features.csv
"""

import argparse
import os

import pandas as pd


def build_team_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """One row per (team, GW) with that team's goals for/against in that
    gameweek, plus a rolling average of goals *conceded* over the previous
    `window` gameweeks (shifted, so it never includes the current GW)."""
    df = df.copy()
    df["goals_for"] = df.apply(
        lambda r: r["team_h_score"] if r["was_home"] else r["team_a_score"], axis=1
    )
    df["goals_against"] = df.apply(
        lambda r: r["team_a_score"] if r["was_home"] else r["team_h_score"], axis=1
    )

    team_gw = (
        df.groupby(["team", "GW"], as_index=False)
        .agg(goals_for=("goals_for", "first"), goals_against=("goals_against", "first"))
        .sort_values(["team", "GW"])
    )

    team_gw["team_goals_conceded_form"] = (
        team_gw.groupby("team")["goals_against"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    team_gw["team_goals_scored_form"] = (
        team_gw.groupby("team")["goals_for"]
        .apply(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        .reset_index(level=0, drop=True)
    )
    return team_gw[["team", "GW", "team_goals_conceded_form", "team_goals_scored_form"]]


def build_player_form(df: pd.DataFrame, window: int) -> pd.DataFrame:
    df = df.sort_values(["name", "GW"]).copy()
    grp = df.groupby("name")

    df["form_points"] = grp["total_points"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df["form_minutes"] = grp["minutes"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    df["form_ict"] = grp["ict_index"].apply(
        lambda s: s.shift(1).rolling(window, min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    # Points from exactly the previous gameweek (no averaging) - feeds a
    # "naive momentum" formula: captain whoever just had a big game.
    df["last_gw_points"] = grp["total_points"].apply(
        lambda s: s.shift(1)
    ).reset_index(level=0, drop=True)

    # Season-long reliability signal: average points across every prior
    # gameweek this season (expanding window, not just the last N).
    df["season_form_points"] = grp["total_points"].apply(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    ).reset_index(level=0, drop=True)

    return df


DEFAULT_RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DEFAULT_OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def build_features(
    season: str,
    window: int = 4,
    raw_dir: str = DEFAULT_RAW_DIR,
    out_dir: str = DEFAULT_OUT_DIR,
) -> str:
    """Full preprocessing pipeline for one season. Returns the path of the
    written feature file. Importable (used by the dashboard's Data Manager
    page) as well as runnable via the CLI below."""
    raw_path = os.path.join(raw_dir, season, "merged_gw.csv")
    df = pd.read_csv(raw_path)

    # was_home sometimes arrives as string "True"/"False" from CSV
    if df["was_home"].dtype == object:
        df["was_home"] = df["was_home"].map({"True": True, "False": False})

    # Official FPL fixture difficulty (1-5, set by FPL before each gameweek -
    # genuinely known ahead of time, so safe to use). Joined via the exact
    # `fixture` id rather than name-matching, so there's no ambiguity.
    fixtures_path = os.path.join(raw_dir, season, "fixtures.csv")
    if os.path.exists(fixtures_path) and "fixture" in df.columns:
        fixtures = pd.read_csv(fixtures_path)[
            ["id", "team_h_difficulty", "team_a_difficulty"]
        ]
        df = df.merge(fixtures, left_on="fixture", right_on="id", how="left")
        df["fixture_difficulty"] = df.apply(
            lambda r: r["team_h_difficulty"] if r["was_home"] else r["team_a_difficulty"],
            axis=1,
        )
    else:
        df["fixture_difficulty"] = pd.NA

    # In merged_gw.csv, `opponent_team` is a numeric team id, while `team` is
    # the player's own team *name*. Map the id to a name via teams.csv so we
    # can join opponent form onto the same team-name space.
    teams_path = os.path.join(raw_dir, season, "teams.csv")
    if os.path.exists(teams_path) and pd.api.types.is_numeric_dtype(df["opponent_team"]):
        teams = pd.read_csv(teams_path)
        # teams.csv columns vary slightly by season; try common id/name pairs
        id_col = "id" if "id" in teams.columns else "team"
        name_col = "name" if "name" in teams.columns else "team_name"
        id_to_name = dict(zip(teams[id_col], teams[name_col]))
        df["opponent_team_name"] = df["opponent_team"].map(id_to_name)
    else:
        df["opponent_team_name"] = df["opponent_team"]

    team_form = build_team_form(df, window)
    df = build_player_form(df, window)

    # Join opponent's defensive/attacking form (as of the gameweek BEFORE this fixture)
    opp_form = team_form.rename(
        columns={
            "team": "opponent_team_name",
            "team_goals_conceded_form": "opp_goals_conceded_form",
            "team_goals_scored_form": "opp_goals_scored_form",
        }
    )
    df = df.merge(
        opp_form,
        on=["opponent_team_name", "GW"],
        how="left",
    )

    keep_cols = [
        "name", "position", "team", "opponent_team", "was_home", "GW",
        "value", "minutes", "total_points",
        "form_points", "form_minutes", "form_ict",
        "last_gw_points", "season_form_points",
        "opp_goals_conceded_form", "opp_goals_scored_form",
        "fixture_difficulty",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    out = df[keep_cols].copy()

    season_out_dir = os.path.join(out_dir, season)
    os.makedirs(season_out_dir, exist_ok=True)
    out_path = os.path.join(season_out_dir, "player_gw_features.csv")
    out.to_csv(out_path, index=False)
    print(f"Wrote {len(out):,} rows to {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--window", type=int, default=4, help="rolling form window, in gameweeks")
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    build_features(args.season, args.window, args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
