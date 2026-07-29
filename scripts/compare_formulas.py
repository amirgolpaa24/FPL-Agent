"""
Run every candidate formula from baseline_heuristic.py against every season
we have processed data for, and rank them.

Usage:
    python3 compare_formulas.py
    python3 compare_formulas.py --seasons 2022-23,2023-24,2024-25
    python3 compare_formulas.py --include-synthetic   # also test test-season

Reads:  ../data/processed/<season>/player_gw_features.csv for each season
Writes: ../data/processed/formula_comparison.csv (one row per formula x season)
Prints: per-season breakdown + an aggregate ranking across all seasons tested
"""

import argparse
import os

import pandas as pd

from baseline_heuristic import FORMULAS
from backtest import run_backtest

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def discover_seasons(include_synthetic: bool) -> list[str]:
    if not os.path.isdir(DATA_DIR):
        return []
    seasons = [
        d for d in sorted(os.listdir(DATA_DIR))
        if os.path.isdir(os.path.join(DATA_DIR, d))
        and os.path.exists(os.path.join(DATA_DIR, d, "player_gw_features.csv"))
    ]
    if not include_synthetic:
        seasons = [s for s in seasons if s != "test-season"]
    return seasons


def evaluate(features: pd.DataFrame, formula: str) -> dict | None:
    results = run_backtest(features, formula=formula)
    if results.empty:
        return None

    total_heuristic = results["heuristic_captain_pts_doubled"].sum()
    total_best = results["best_possible_pts_doubled"].sum()
    total_field = (results["field_average_pts"] * 2).sum()
    n_gw = len(results)
    hit_rate = (results["heuristic_captain"] == results["best_possible_captain"]).mean()
    gap_capture = (
        (total_heuristic - total_field) / (total_best - total_field)
        if total_best != total_field
        else float("nan")
    )

    return {
        "n_gw": n_gw,
        "total_pts": total_heuristic,
        "avg_pts_per_gw": total_heuristic / n_gw,
        "hit_rate": hit_rate,
        "gap_capture": gap_capture,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons", default=None,
        help="Comma-separated season list. Default: auto-detect every season under data/processed/",
    )
    parser.add_argument(
        "--include-synthetic", action="store_true",
        help="Also include the synthetic test-season (excluded by default)",
    )
    args = parser.parse_args()

    seasons = (
        args.seasons.split(",") if args.seasons
        else discover_seasons(args.include_synthetic)
    )
    if not seasons:
        print("No processed seasons found. Run build_dataset.py for at least one season first.")
        return

    print(f"Comparing {len(FORMULAS)} formulas across {len(seasons)} season(s): {', '.join(seasons)}\n")

    rows = []
    for season in seasons:
        features_path = os.path.join(DATA_DIR, season, "player_gw_features.csv")
        if not os.path.exists(features_path):
            print(f"  skipping {season}: no player_gw_features.csv")
            continue
        features = pd.read_csv(features_path)

        for formula_name in FORMULAS:
            metrics = evaluate(features, formula_name)
            if metrics is None:
                continue
            rows.append({"season": season, "formula": formula_name, **metrics})

    comparison = pd.DataFrame(rows)
    out_path = os.path.join(DATA_DIR, "formula_comparison.csv")
    comparison.to_csv(out_path, index=False)
    print(f"Wrote {len(comparison)} rows to {out_path}\n")

    print("=== Per season x formula ===")
    print(
        comparison.assign(
            hit_rate=lambda d: (d["hit_rate"] * 100).round(1),
            gap_capture=lambda d: (d["gap_capture"] * 100).round(1),
            avg_pts_per_gw=lambda d: d["avg_pts_per_gw"].round(2),
        ).to_string(index=False)
    )

    print("\n=== Aggregate across all seasons tested (ranked by average gap-capture) ===")
    agg = (
        comparison.groupby("formula", as_index=False)
        .agg(
            seasons_tested=("season", "nunique"),
            avg_gap_capture=("gap_capture", "mean"),
            avg_hit_rate=("hit_rate", "mean"),
            avg_pts_per_gw=("avg_pts_per_gw", "mean"),
            total_pts=("total_pts", "sum"),
        )
        .sort_values("avg_gap_capture", ascending=False)
    )
    agg["avg_gap_capture"] = (agg["avg_gap_capture"] * 100).round(1)
    agg["avg_hit_rate"] = (agg["avg_hit_rate"] * 100).round(1)
    agg["avg_pts_per_gw"] = agg["avg_pts_per_gw"].round(2)
    print(agg.to_string(index=False))

    best = agg.iloc[0]
    print(f"\nBest formula overall: {best['formula']} "
          f"({best['avg_gap_capture']:.1f}% avg gap-capture across {best['seasons_tested']} season(s))")


if __name__ == "__main__":
    main()
