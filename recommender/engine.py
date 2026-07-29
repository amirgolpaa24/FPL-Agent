"""
Deterministic, stats-based decision engine. This is the "solid" mode - no LLM,
just the formulas that actually won our backtests.

Every function takes plain PlayerFeatures / Squad objects and returns plain
dicts, so it's trivially testable and JSON-serializable.

Design choices grounded in the backtesting we did (scripts/compare_formulas.py):
  - Default scoring formula is `expanding_season_form`, which had the best
    average gap-capture across five seasons - season-long consistency beats a
    short recent-form window for FPL.
  - "Eligibility" (regular minutes) matters: a player with a big past game but
    who's stopped starting is a trap. We surface this rather than silently
    dropping players.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import pandas as pd

# Reuse the exact formulas validated by the backtests as the single source of
# truth for "expected points" scoring.
_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from baseline_heuristic import FORMULAS, MIN_MINUTES_FORM  # noqa: E402

DEFAULT_SCORING_FORMULA = "composite"

# Valid FPL starting-XI formation constraints (GK always exactly 1).
FORMATION_BOUNDS = {"GK": (1, 1), "DEF": (3, 5), "MID": (2, 5), "FWD": (1, 3)}
STARTING_XI_SIZE = 11


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def expected_points(
    players: list,
    formula: str = DEFAULT_SCORING_FORMULA,
) -> pd.DataFrame:
    """Score a list of PlayerFeatures with a backtested formula. Returns a
    DataFrame with an `expected_score` column, preserving input order via
    a stable index. Does NOT filter by minutes - callers decide that."""
    if formula not in FORMULAS:
        raise ValueError(f"Unknown formula {formula!r}. Options: {list(FORMULAS)}")
    if not players:
        return pd.DataFrame()

    df = pd.DataFrame([p.to_row() for p in players])
    # Coerce numeric feature columns to real numeric dtype. When a whole column
    # is None (e.g. availability in historical data), pandas infers object
    # dtype, which makes .fillna() noisy and fragile - this keeps scoring clean.
    numeric_cols = [
        "form_points", "season_form_points", "last_gw_points", "form_ict",
        "form_minutes", "fixture_difficulty", "opp_goals_conceded_form",
        "availability", "fpl_ep_next", "price",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["expected_score"] = FORMULAS[formula](df)
    df["eligible"] = df["form_minutes"].fillna(0) >= MIN_MINUTES_FORM
    return df


def _attach(df: pd.DataFrame, players: list) -> pd.DataFrame:
    """Re-attach the original PlayerFeatures objects to a scored frame."""
    df = df.copy()
    df["_obj"] = players
    return df


# ---------------------------------------------------------------------------
# captain
# ---------------------------------------------------------------------------

def stats_captain(
    candidates: list,
    formula: str = DEFAULT_SCORING_FORMULA,
    top_n: int = 5,
) -> dict:
    """Rank captain candidates by expected points. Only regularly-playing
    players can be recommended; the rest are still scored but flagged."""
    if not candidates:
        return {"recommendation": None, "ranked": [], "note": "no candidates supplied"}

    df = _attach(expected_points(candidates, formula), candidates)
    df = df.sort_values("expected_score", ascending=False)

    eligible = df[df["eligible"]]
    pick = (eligible.iloc[0] if not eligible.empty else df.iloc[0])

    ranked = [
        {
            "name": r["name"],
            "team": r["team"],
            "position": r["position"],
            "expected_score": round(float(r["expected_score"]), 3),
            "eligible": bool(r["eligible"]),
            "form_points": _num(r.get("form_points")),
            "season_form_points": _num(r.get("season_form_points")),
            "fixture_difficulty": _num(r.get("fixture_difficulty")),
        }
        for _, r in df.head(top_n).iterrows()
    ]

    return {
        "recommendation": pick["name"],
        "recommendation_team": pick["team"],
        "expected_score": round(float(pick["expected_score"]), 3),
        "formula": formula,
        "ranked": ranked,
        "reasoning": (
            f"{pick['name']} has the highest expected score ({pick['expected_score']:.2f}) "
            f"among regularly-playing candidates, driven mostly by a season-long scoring "
            f"average of {_fmt(pick.get('season_form_points'))} and a fixture difficulty of "
            f"{_fmt(pick.get('fixture_difficulty'))}/5."
        ),
    }


# ---------------------------------------------------------------------------
# starting XI
# ---------------------------------------------------------------------------

def stats_lineup(
    squad,
    formula: str = DEFAULT_SCORING_FORMULA,
) -> dict:
    """Choose the starting XI + bench + captain from a 15-man squad by
    maximizing total expected points over all legal formations."""
    players = squad.players if hasattr(squad, "players") else squad
    if not players:
        return {"error": "empty squad"}

    df = _attach(expected_points(players, formula), players)

    by_pos = {pos: df[df["position"] == pos].sort_values("expected_score", ascending=False)
              for pos in FORMATION_BOUNDS}

    # Enumerate legal outfield formations (GK fixed at 1) and pick the best.
    best = None
    for n_def in range(*_r(FORMATION_BOUNDS["DEF"])):
        for n_mid in range(*_r(FORMATION_BOUNDS["MID"])):
            for n_fwd in range(*_r(FORMATION_BOUNDS["FWD"])):
                if 1 + n_def + n_mid + n_fwd != STARTING_XI_SIZE:
                    continue
                counts = {"GK": 1, "DEF": n_def, "MID": n_mid, "FWD": n_fwd}
                if any(len(by_pos[p]) < c for p, c in counts.items()):
                    continue
                total = sum(by_pos[p].head(c)["expected_score"].sum() for p, c in counts.items())
                if best is None or total > best["total"]:
                    best = {"total": total, "counts": counts}

    if best is None:
        return {"error": "no legal formation possible from this squad "
                         "(check squad composition)"}

    counts = best["counts"]
    starters_idx = []
    for pos, c in counts.items():
        starters_idx.extend(by_pos[pos].head(c).index.tolist())

    starters = df.loc[starters_idx].sort_values("expected_score", ascending=False)
    bench = df.drop(index=starters_idx).sort_values("expected_score", ascending=False)

    captain = starters.iloc[0]
    vice = starters.iloc[1] if len(starters) > 1 else captain

    return {
        "formation": f"{counts['DEF']}-{counts['MID']}-{counts['FWD']}",
        "expected_total": round(float(best["total"]), 2),
        "formula": formula,
        "starting_xi": [_player_line(r) for _, r in starters.iterrows()],
        "bench": [_player_line(r) for _, r in bench.iterrows()],
        "captain": captain["name"],
        "vice_captain": vice["name"],
        "reasoning": (
            f"The {counts['DEF']}-{counts['MID']}-{counts['FWD']} formation maximizes total "
            f"expected points ({best['total']:.1f}) across all legal shapes. "
            f"{captain['name']} is the highest-scoring starter, so captains; "
            f"{vice['name']} is vice."
        ),
    }


# ---------------------------------------------------------------------------
# initial squad selection
# ---------------------------------------------------------------------------

SQUAD_QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_CLUB = 3
SQUAD_SIZE = 15


def stats_squad(
    pool: list,
    budget: float = 100.0,
    formula: str = DEFAULT_SCORING_FORMULA,
    min_minutes: float = MIN_MINUTES_FORM,
    prefer_names: set | None = None,
) -> dict:
    """Build a full legal 15-man squad from a player pool.

    Greedy selection by expected points, always reserving enough budget to
    fill every remaining slot with the cheapest legal players - so the result
    is always a valid £`budget`m squad (2 GK / 5 DEF / 5 MID / 3 FWD, max 3
    per club). Because premium players force cheap fills elsewhere, this
    naturally yields the standard FPL shape: money on a strong starting XI,
    cheap bench/backup keeper.

    `prefer_names`: names to pull in first when legal (used to render an AI's
    proposed squad - the AI's picks are prioritised, and the rest of the slots
    are filled with the best legal, affordable players). The result is always
    a legal squad, so it can always be shown on the pitch.

    This is a strong heuristic, not a proven global optimum (that would need an
    ILP solver). After building, the best XI / captain / formation are chosen
    by the same optimizer the Starting XI feature uses.
    """
    if not pool:
        return {"error": "empty player pool"}

    scored = _attach(expected_points(pool, formula), pool)
    # Sort by a preference-adjusted key (big bonus for preferred players) but
    # keep the real expected_score untouched for display.
    prefer = {n.strip().lower() for n in (prefer_names or set())}
    scored["_sort"] = scored["expected_score"] + scored["name"].apply(
        lambda n: 1e6 if str(n).strip().lower() in prefer else 0.0
    )
    scored = scored.sort_values("_sort", ascending=False)
    rows = scored.to_dict("records")  # each has name/position/team/price/_obj/...

    quota = dict(SQUAD_QUOTA)
    club_count: dict[str, int] = {}
    chosen_objs = []
    chosen_names: set[str] = set()
    spent = 0.0

    def min_cost_to_fill(remaining: dict, exclude_names: set) -> float:
        total = 0.0
        for pos, need in remaining.items():
            if need <= 0:
                continue
            prices = sorted(
                r["price"] for r in rows
                if r["position"] == pos and r["name"] not in exclude_names
            )
            if len(prices) < need:
                return float("inf")
            total += sum(prices[:need])
        return total

    for r in rows:
        pos, club, price, name = r["position"], r["team"], r["price"], r["name"]
        if quota.get(pos, 0) <= 0 or name in chosen_names:
            continue
        if club_count.get(club, 0) >= MAX_PER_CLUB:
            continue
        remaining = {p: (n - 1 if p == pos else n) for p, n in quota.items()}
        if spent + price + min_cost_to_fill(remaining, chosen_names | {name}) > budget + 1e-6:
            continue
        chosen_objs.append(r["_obj"])
        chosen_names.add(name)
        spent += price
        quota[pos] -= 1
        club_count[club] = club_count.get(club, 0) + 1
        if all(v == 0 for v in quota.values()):
            break

    if any(v > 0 for v in quota.values()):
        return {"error": "could not assemble a legal squad within budget "
                         f"(£{budget}m). Remaining slots: "
                         f"{{k: v for k, v in quota.items() if v > 0}}"}

    # Import here to avoid a circular reference at module load.
    from .features import Squad as _Squad
    squad_obj = _Squad(players=chosen_objs, bank=round(budget - spent, 1))
    lineup = stats_lineup(squad_obj, formula=formula)

    squad_rows = _attach(expected_points(chosen_objs, formula), chosen_objs)
    squad_rows = squad_rows.sort_values(
        ["position", "expected_score"], ascending=[True, False]
    )

    return {
        "squad": [_player_line(r) for _, r in squad_rows.iterrows()],
        "total_cost": round(spent, 1),
        "budget": budget,
        "bank_left": round(budget - spent, 1),
        "formula": formula,
        "best_xi": lineup.get("starting_xi", []),
        "bench": lineup.get("bench", []),
        "formation": lineup.get("formation"),
        "captain": lineup.get("captain"),
        "vice_captain": lineup.get("vice_captain"),
        "expected_xi_points": lineup.get("expected_total"),
        "reasoning": (
            f"Assembled a £{spent:.1f}m squad maximising expected points greedily "
            f"under the £{budget:.0f}m budget, then set the best legal XI "
            f"({lineup.get('formation')}) with {lineup.get('captain')} as captain."
        ),
    }


# ---------------------------------------------------------------------------
# transfers
# ---------------------------------------------------------------------------

def stats_transfers(
    squad,
    candidate_pool: list,
    formula: str = DEFAULT_SCORING_FORMULA,
    max_suggestions: int = 5,
    hit_cost: int = 4,
) -> dict:
    """Suggest single-player swaps that raise expected points. Respects
    budget (bank + selling price), position matching, and the max-3-per-club
    rule. Each suggestion's net gain accounts for the -`hit_cost` penalty when
    it would exceed the free-transfer allowance."""
    squad_players = squad.players
    bank = squad.bank
    free = squad.free_transfers

    if not candidate_pool:
        return {"suggestions": [], "note": "empty candidate pool"}

    squad_scored = _attach(expected_points(squad_players, formula), squad_players)
    pool_scored = _attach(expected_points(candidate_pool, formula), candidate_pool)

    squad_names = {p.name for p in squad_players}
    pool_scored = pool_scored[~pool_scored["name"].isin(squad_names)]

    club_counts = {}
    for p in squad_players:
        club_counts[p.team] = club_counts.get(p.team, 0) + 1

    suggestions = []
    for _, out_row in squad_scored.iterrows():
        pos = out_row["position"]
        out_price = out_row["price"]
        # candidates of same position, affordable given bank + what we free up
        budget = bank + out_price
        options = pool_scored[
            (pool_scored["position"] == pos)
            & (pool_scored["price"] <= budget + 1e-9)
            & (pool_scored["eligible"])
        ]
        for _, in_row in options.iterrows():
            # club constraint: bringing in_row in, taking out_row out
            new_club_count = club_counts.get(in_row["team"], 0) + (0 if in_row["team"] == out_row["team"] else 1)
            if in_row["team"] != out_row["team"] and new_club_count > 3:
                continue
            gain = float(in_row["expected_score"] - out_row["expected_score"])
            if gain <= 0:
                continue
            suggestions.append({
                "out": out_row["name"], "out_team": out_row["team"],
                "out_expected": round(float(out_row["expected_score"]), 3),
                "in": in_row["name"], "in_team": in_row["team"],
                "in_expected": round(float(in_row["expected_score"]), 3),
                "position": pos,
                "price_change": round(float(in_row["price"] - out_price), 1),
                "raw_gain": round(gain, 3),
            })

    suggestions.sort(key=lambda s: s["raw_gain"], reverse=True)

    # Keep the best distinct-out suggestions, and compute net gain after hits.
    seen_out = set()
    deduped = []
    for s in suggestions:
        if s["out"] in seen_out:
            continue
        seen_out.add(s["out"])
        n_transfer = len(deduped) + 1
        hit = 0 if n_transfer <= free else hit_cost
        s = dict(s, transfer_index=n_transfer, hit_applied=hit,
                 net_gain=round(s["raw_gain"] - hit, 3))
        deduped.append(s)
        if len(deduped) >= max_suggestions:
            break

    positive = [s for s in deduped if s["net_gain"] > 0]
    return {
        "suggestions": deduped,
        "recommended_transfers": positive,
        "formula": formula,
        "bank": bank,
        "free_transfers": free,
        "reasoning": (
            f"Ranked single swaps by expected-point gain. {len(positive)} clear "
            f"buy(s) after accounting for the -{hit_cost} hit beyond {free} free "
            f"transfer(s)."
            if positive else
            f"No swap beats keeping your squad once the -{hit_cost} hit is "
            f"considered - holding is the stats-optimal move."
        ),
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _r(bounds: tuple[int, int]) -> tuple[int, int]:
    return (bounds[0], bounds[1] + 1)


def _num(v) -> Optional[float]:
    try:
        if v is None or pd.isna(v):
            return None
        return round(float(v), 3)
    except (TypeError, ValueError):
        return None


def _fmt(v) -> str:
    n = _num(v)
    return "n/a" if n is None else f"{n:.2f}"


def _player_line(r) -> dict:
    code = r.get("team_code") if hasattr(r, "get") else r["team_code"]
    return {
        "name": r["name"],
        "team": r["team"],
        "team_code": (int(code) if code is not None and not pd.isna(code) else None),
        "position": r["position"],
        "price": round(float(r["price"]), 1),
        "expected_score": round(float(r["expected_score"]), 3),
        "eligible": bool(r["eligible"]),
    }
