"""
Shared helpers for the FPL Agent multipage Streamlit dashboard.
Every page imports from here so data loading, caching, and formula metadata
stay consistent across the app.
"""

import os
import sys

import pandas as pd
import streamlit as st

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")
SCRIPTS_DIR = os.path.join(ROOT_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from baseline_heuristic import FORMULAS, DEFAULT_FORMULA, MIN_MINUTES_FORM  # noqa: E402,F401
from backtest import run_backtest  # noqa: E402

FORMULA_LABELS = {
    "fpl_official": "FPL official expected points (ep_next)",
    "composite": "Composite multi-factor model (heuristic)",
    "form_fdr": "Form + official FDR (baseline)",
    "form_proxy_only": "Form + goals-conceded proxy",
    "form_blended_fixture": "Form + blended fixture signal",
    "ict_fixture": "ICT index + fixture",
    "momentum_last_gw": "Momentum (last GW only)",
    "expanding_season_form": "Season-long expanding form",
    "form_fdr_home_adjusted": "Form + FDR + home adjustment",
}
FORMULA_NAMES = list(FORMULAS.keys())

# Pre-gameweek features an AI/model could legitimately use, with readable names.
PREDICTIVE_FEATURES = {
    "form_points": "Recent points form (4-GW avg)",
    "season_form_points": "Season-long points average",
    "last_gw_points": "Last gameweek's points only",
    "form_ict": "Recent ICT index (4-GW avg)",
    "form_minutes": "Recent minutes (4-GW avg)",
    "fixture_difficulty": "Official fixture difficulty (FDR)",
    "opp_goals_conceded_form": "Opponent's goals-conceded form",
    "value": "Player price",
}

SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3


def label(formula: str) -> str:
    return FORMULA_LABELS.get(formula, formula)


def formula_doc(formula: str) -> str:
    return (FORMULAS[formula].__doc__ or "").strip()


def _features_path(season: str) -> str:
    return os.path.join(DATA_DIR, season, "player_gw_features.csv")


def list_processed_seasons(exclude_synthetic: bool = True) -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    seasons = sorted(
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and os.path.exists(_features_path(d))
    )
    if exclude_synthetic:
        seasons = [s for s in seasons if s != "test-season"]
    return seasons


def list_raw_seasons() -> list[str]:
    if not os.path.isdir(RAW_DIR):
        return []
    return sorted(
        d for d in os.listdir(RAW_DIR)
        if os.path.isdir(os.path.join(RAW_DIR, d))
        and os.path.exists(os.path.join(RAW_DIR, d, "merged_gw.csv"))
    )


@st.cache_data
def _load_features(season: str, mtime: float) -> pd.DataFrame:
    # mtime is part of the cache key on purpose: reprocessing a season on the
    # Data Manager page changes the file's mtime, which automatically busts
    # this cache - no manual "Clear cache" needed.
    return pd.read_csv(_features_path(season))


def load_features(season: str) -> pd.DataFrame:
    return _load_features(season, os.path.getmtime(_features_path(season)))


@st.cache_data
def _compute_backtest(season: str, formula: str, mtime: float) -> pd.DataFrame:
    features = _load_features(season, mtime)
    return run_backtest(features, formula=formula)


def compute_backtest(season: str, formula: str) -> pd.DataFrame:
    return _compute_backtest(season, formula, os.path.getmtime(_features_path(season)))


def summarize(results: pd.DataFrame) -> dict | None:
    if results is None or results.empty:
        return None
    total_heuristic = results["heuristic_captain_pts_doubled"].sum()
    total_best = results["best_possible_pts_doubled"].sum()
    total_field = (results["field_average_pts"] * 2).sum()
    n_gw = len(results)
    hit_rate = (results["heuristic_captain"] == results["best_possible_captain"]).mean()
    gap_capture = (
        (total_heuristic - total_field) / (total_best - total_field)
        if total_best != total_field else float("nan")
    )
    return dict(
        n_gw=n_gw, total_heuristic=total_heuristic, total_best=total_best,
        total_field=total_field, hit_rate=hit_rate, gap_capture=gap_capture,
    )


def season_picker(key: str = "season") -> str | None:
    """Consistent season selector for pages that operate on one season."""
    seasons = list_processed_seasons()
    if not seasons:
        st.error(
            "No processed seasons found. Head to the **Data Manager** page to "
            "download and preprocess at least one season."
        )
        return None
    return st.sidebar.selectbox(
        "Season", seasons, index=len(seasons) - 1, key=key,
        help="Which season's data this page analyzes.",
    )
