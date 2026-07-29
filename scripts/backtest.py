"""
Backtest the baseline captain heuristic across a full season.

For every gameweek (once enough history exists to compute form), asks
baseline_heuristic to pick a captain, then looks up what that player actually
scored. Compares against:
  - best_possible: the single best actual score among eligible players that GW
    (the ceiling - what a perfect-hindsight captain would have scored)
  - field_average: average actual score among eligible players that GW
    (the floor - what a random reasonable pick would average)

Captain points are doubled, matching real FPL scoring.

Usage:
    python3 backtest.py --season 2024-25 --formula form_fdr

Run with --formula list to see all available formulas (defined in
baseline_heuristic.py). To compare every formula across every season at
once, use compare_formulas.py instead.

Reads:  ../data/processed/<season>/player_gw_features.csv
Writes: ../data/processed/<season>/backtest_results.csv (per-gameweek detail)
Prints: season-level summary
"""

import argparse
import os
import sys

import pandas as pd

from baseline_heuristic import score_gameweek, FORMULAS, DEFAULT_FORMULA, MIN_MINUTES_FORM


def run_backtest(features: pd.DataFrame, formula: str = DEFAULT_FORMULA) -> pd.DataFrame:
    rows = []
    for gw, gw_df in features.groupby("GW"):
        scored = score_gameweek(gw_df, formula=formula)
        if scored.empty:
            continue

        captain = scored.iloc[0]
        eligible = gw_df[gw_df["form_minutes"].fillna(0) >= MIN_MINUTES_FORM]
        if eligible.empty:
            continue

        best_row = eligible.loc[eligible["total_points"].idxmax()]

        rows.append(
            {
                "GW": gw,
                "heuristic_captain": captain["name"],
                "heuristic_captain_pts": captain["total_points"],
                "heuristic_captain_pts_doubled": captain["total_points"] * 2,
                "best_possible_captain": best_row["name"],
                "best_possible_pts": best_row["total_points"],
                "best_possible_pts_doubled": best_row["total_points"] * 2,
                "field_average_pts": eligible["total_points"].mean(),
                "eligible_pool_size": len(eligible),
            }
        )
    return pd.DataFrame(rows)


def summarize(results: pd.DataFrame) -> None:
    if results.empty:
        print("No gameweeks had enough history to backtest.")
        return

    total_heuristic = results["heuristic_captain_pts_doubled"].sum()
    total_best = results["best_possible_pts_doubled"].sum()
    total_field = (results["field_average_pts"] * 2).sum()
    n_gw = len(results)
    hit_rate = (
        results["heuristic_captain"] == results["best_possible_captain"]
    ).mean()

    print(f"Gameweeks backtested: {n_gw}")
    print(f"Heuristic captain total points (doubled): {total_heuristic:.0f}  "
          f"(avg {total_heuristic / n_gw:.1f}/GW)")
    print(f"Best-possible captain total points (doubled): {total_best:.0f}  "
          f"(avg {total_best / n_gw:.1f}/GW)")
    print(f"Field-average captain total points (doubled): {total_field:.0f}  "
          f"(avg {total_field / n_gw:.1f}/GW)")
    print(f"Exact hit rate (heuristic picked the actual best captain): {hit_rate:.1%}")
    print(
        f"Heuristic captures {100 * (total_heuristic - total_field) / (total_best - total_field):.0f}% "
        f"of the gap between a random-ish pick and a perfect-hindsight pick."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument(
        "--formula", default=DEFAULT_FORMULA,
        help=f"Which scoring formula to use, or 'list' to show options. Choices: {list(FORMULAS)}",
    )
    parser.add_argument(
        "--data-dir", default=os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    )
    args = parser.parse_args()

    if args.formula == "list":
        for name, fn in FORMULAS.items():
            print(f"{name}: {(fn.__doc__ or '').strip().splitlines()[0]}")
        sys.exit(0)

    features_path = os.path.join(args.data_dir, args.season, "player_gw_features.csv")
    features = pd.read_csv(features_path)

    results = run_backtest(features, formula=args.formula)
    out_path = os.path.join(args.data_dir, args.season, "backtest_results.csv")
    results.to_csv(out_path, index=False)
    print(f"Wrote per-gameweek results to {out_path}  (formula: {args.formula})\n")

    summarize(results)


if __name__ == "__main__":
    main()
