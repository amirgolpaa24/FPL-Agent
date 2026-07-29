"""Backtest strategies for the season's biggest single decision: picking the
initial 15-man squad with only prior-season knowledge."""

import pandas as pd
import plotly.express as px
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="Squad Builder - FPL Agent", layout="wide", page_icon="🧢")

st.title("🧢 Squad Builder")
st.markdown(
    "Before gameweek 1 you know nothing about the new season - only last season's "
    "numbers and this season's prices. This page builds squads using **only that**, "
    "then scores them against what actually happened, so squad-selection strategies "
    "get the same honest backtest treatment as captaincy formulas."
)

seasons = lib.list_processed_seasons()
if len(seasons) < 2:
    st.error(
        "Squad building needs at least two consecutive processed seasons (one for "
        "prior knowledge, one to evaluate against). Add more on the Data Manager page."
    )
    st.stop()

target_season = st.selectbox(
    "Season to build for", seasons[1:], index=len(seasons) - 2,
    help="The squad is picked as of this season's GW1, knowing only the previous "
    "season's performance and this season's starting prices.",
)
prior_season = seasons[seasons.index(target_season) - 1]
st.caption(f"Prior knowledge: **{prior_season}** · Evaluated against: **{target_season}**")

budget_m = st.slider(
    "Budget (£m)", 80.0, 110.0, 100.0, 0.5,
    help="Real FPL gives you £100m. Lower it to see how strategies cope when "
    "premium players are out of reach.",
)

# ---------------------------------------------------------------- data prep
target = lib.load_features(target_season)
prior = lib.load_features(prior_season)

prior_totals = (
    prior.groupby("name")
    .agg(prior_points=("total_points", "sum"), prior_minutes=("minutes", "sum"))
    .reset_index()
)

start_prices = (
    target.sort_values("GW").groupby("name")
    .agg(price=("value", "first"), team=("team", "first"),
         position=("position", "first"), first_gw=("GW", "first"))
    .reset_index()
)

actual_totals = (
    target.groupby("name")["total_points"].sum().rename("actual_points").reset_index()
)

pool = (
    start_prices.merge(prior_totals, on="name", how="inner")
    .merge(actual_totals, on="name", how="left")
)
pool = pool[pool["prior_minutes"] >= 900]  # ~10 full games last season
pool["price_m"] = pool["price"] / 10
pool["prior_ppm"] = pool["prior_points"] / pool["price_m"]
pool["actual_points"] = pool["actual_points"].fillna(0)

n_new = len(start_prices) - len(pool)
st.caption(
    f"Candidate pool: {len(pool)} players with meaningful minutes in {prior_season}. "
    f"({n_new} players excluded: new signings/promoted-team players with no prior-season "
    f"record - exactly the blind spot a human or AI would face too.)"
)


# ---------------------------------------------------------------- greedy picker
def build_squad(pool: pd.DataFrame, score_col: str, budget: float) -> pd.DataFrame | None:
    """Greedy squad selection under FPL rules: 2 GK / 5 DEF / 5 MID / 3 FWD,
    max 3 per club, total cost <= budget. Picks best-scoring available player
    that still leaves enough budget to fill remaining slots with the cheapest
    legal options."""
    candidates = pool.sort_values(score_col, ascending=False).copy()
    quota = dict(lib.SQUAD_QUOTA)
    club_count: dict[str, int] = {}
    picked_idx: list[int] = []
    spent = 0.0

    def min_cost_to_fill(remaining_quota: dict, exclude_idx: list[int]) -> float:
        total = 0.0
        avail = candidates[~candidates.index.isin(exclude_idx)]
        for pos, need in remaining_quota.items():
            if need <= 0:
                continue
            cheapest = avail[avail["position"] == pos].nsmallest(need, "price_m")["price_m"]
            if len(cheapest) < need:
                return float("inf")
            total += cheapest.sum()
        return total

    for idx, row in candidates.iterrows():
        pos, club, cost = row["position"], row["team"], row["price_m"]
        if quota.get(pos, 0) <= 0:
            continue
        if club_count.get(club, 0) >= lib.MAX_PER_CLUB:
            continue
        remaining = {p: (n - 1 if p == pos else n) for p, n in quota.items()}
        exclude = picked_idx + [idx]
        if spent + cost + min_cost_to_fill(remaining, exclude) > budget:
            continue
        picked_idx.append(idx)
        spent += cost
        quota[pos] -= 1
        club_count[club] = club_count.get(club, 0) + 1
        if all(v == 0 for v in quota.values()):
            break

    if any(v > 0 for v in quota.values()):
        return None
    return candidates.loc[picked_idx]


STRATEGIES = {
    "Proven points": (
        "prior_points",
        "Pick the highest prior-season scorers you can afford - premium-heavy.",
    ),
    "Value (pts/£m)": (
        "prior_ppm",
        "Pick the best prior-season points-per-million - budget-heavy, spreads funds.",
    ),
    "Hybrid": (
        "hybrid_score",
        "Balance the two: normalized prior points + normalized value, equally weighted.",
    ),
}

pool["hybrid_score"] = (
    pool["prior_points"] / pool["prior_points"].max()
    + pool["prior_ppm"] / pool["prior_ppm"].max()
)

# ---------------------------------------------------------------- run + compare
st.subheader(
    "Strategy comparison",
    help="Each strategy builds a full legal 15-man squad under the budget. "
    "'Actual points' = what those 15 players really scored across the whole "
    f"target season ({target_season}) - a set-and-forget measure, ignoring "
    "transfers and bench decisions, so strategies are compared like-for-like.",
)

squads = {}
results = []
for strat_name, (score_col, desc) in STRATEGIES.items():
    squad = build_squad(pool, score_col, budget_m)
    if squad is None:
        results.append({"Strategy": strat_name, "Cost (£m)": None, "Actual points": None})
        continue
    squads[strat_name] = squad
    results.append({
        "Strategy": strat_name,
        "Cost (£m)": squad["price_m"].sum(),
        "Prior-season pts": squad["prior_points"].sum(),
        "Actual points": squad["actual_points"].sum(),
    })

results_df = pd.DataFrame(results)
fig_cmp = px.bar(
    results_df.dropna(), x="Strategy", y="Actual points",
    text="Actual points", color="Strategy",
)
fig_cmp.update_traces(texttemplate="%{text:.0f}")
st.plotly_chart(fig_cmp, use_container_width=True)

st.dataframe(
    results_df.round(1), use_container_width=True, hide_index=True,
)

best_strat = results_df.dropna().sort_values("Actual points", ascending=False).iloc[0]["Strategy"]
st.success(
    f"For {target_season}, **{best_strat}** built the best squad. Try other target "
    "seasons above - if the same strategy keeps winning, that's a real signal an AI "
    "squad-picker should start from it."
)

# ---------------------------------------------------------------- squad detail
st.subheader("Inspect a squad")
strat_choice = st.radio(
    "Strategy", list(squads.keys()), horizontal=True,
    help=" · ".join(f"{k}: {v[1]}" for k, v in STRATEGIES.items()),
)
squad = squads[strat_choice].sort_values(
    ["position", "prior_points"], ascending=[True, False]
)
st.dataframe(
    squad[["name", "position", "team", "price_m", "prior_points", "actual_points"]]
    .rename(columns={
        "name": "Player", "position": "Pos", "team": "Team",
        "price_m": "£m", "prior_points": f"Pts {prior_season}",
        "actual_points": f"Pts {target_season}",
    }).round(1),
    use_container_width=True, hide_index=True,
)

c1, c2, c3 = st.columns(3)
c1.metric("Squad cost", f"£{squad['price_m'].sum():.1f}m",
          help=f"Out of the £{budget_m:.0f}m budget.")
c2.metric(f"Prior season ({prior_season})", f"{squad['prior_points'].sum():.0f} pts",
          help="What these 15 scored last season - the information the pick was based on.")
c3.metric(f"Actual ({target_season})", f"{squad['actual_points'].sum():.0f} pts",
          help="What they really scored in the target season - the outcome.")

st.caption(
    "Caveats worth knowing: player names must match across seasons (transfers within "
    "the league are fine; renamed entries aren't), and 'actual points' assumes you "
    "kept all 15 forever - real managers improve on this with transfers, which is "
    "exactly what the Transfer Signals page is about."
)
