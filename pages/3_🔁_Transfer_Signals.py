"""Which metrics actually predict points - the evidence base for transfer
decisions and for choosing what features an AI assistant should look at."""

import pandas as pd
import plotly.express as px
import streamlit as st

import fpl_lib as lib


def spearman(a: pd.Series, b: pd.Series) -> float:
    """Spearman rank correlation without the scipy dependency
    (Pearson correlation of the ranks - mathematically identical)."""
    return a.rank().corr(b.rank())

st.set_page_config(page_title="Transfer Signals - FPL Agent", layout="wide", page_icon="🔁")

st.title("🔁 Transfer Signals")
st.markdown(
    "Transfers are a *prediction* problem: will player X outscore player Y from now "
    "on? This page measures which pre-gameweek signals actually carry predictive "
    "information - the same evidence an AI assistant should weigh."
)

seasons = lib.list_processed_seasons()
if not seasons:
    st.error("No processed seasons - use the Data Manager page first.")
    st.stop()

season = st.selectbox("Season", seasons, index=len(seasons) - 1)
features = lib.load_features(season)
eligible = features[features["form_minutes"].fillna(0) >= lib.MIN_MINUTES_FORM].copy()

tab_pred, tab_scout, tab_value = st.tabs([
    "🔬 Metric predictiveness", "🔍 Transfer scout", "💰 Value & consistency",
])

# ------------------------------------------------------------- predictiveness
with tab_pred:
    st.subheader(
        "How well does each metric predict that gameweek's points?",
        help="Spearman rank correlation between each pre-kickoff metric and the "
        "points actually scored that gameweek, among regularly-playing players. "
        "Higher magnitude = more predictive. Sign shows direction (FDR is "
        "negative because harder fixture → fewer points).",
    )

    corr_rows = []
    for col, nice in lib.PREDICTIVE_FEATURES.items():
        if col not in eligible.columns:
            continue
        sub = eligible.dropna(subset=[col, "total_points"])
        if len(sub) < 100:
            continue
        corr = spearman(sub[col], sub["total_points"])
        corr_rows.append({"Metric": nice, "Spearman correlation": corr})
    corr_df = pd.DataFrame(corr_rows).sort_values("Spearman correlation")

    fig = px.bar(
        corr_df, x="Spearman correlation", y="Metric", orientation="h",
        text=corr_df["Spearman correlation"].map(lambda v: f"{v:+.3f}"),
        color="Spearman correlation", color_continuous_scale="RdYlGn",
        range_color=[-0.3, 0.3],
    )
    fig.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    st.info(
        "Reading this: even the best single metric correlates weakly with next "
        "points - single-gameweek outcomes are mostly noise. That's precisely why "
        "combining signals (and eventually an AI reasoning layer that also reads "
        "news/injuries) matters more than chasing any one stat.",
        icon="💡",
    )

    st.subheader(
        "Does predictiveness hold across seasons?",
        help="Same correlation, computed per season. Stable bars season-to-season "
        "mean the metric is a reliable input; wild swings mean it's situational.",
    )
    multi_rows = []
    for s in seasons:
        f_s = lib.load_features(s)
        el = f_s[f_s["form_minutes"].fillna(0) >= lib.MIN_MINUTES_FORM]
        for col, nice in lib.PREDICTIVE_FEATURES.items():
            if col not in el.columns:
                continue
            sub = el.dropna(subset=[col, "total_points"])
            if len(sub) < 100:
                continue
            multi_rows.append({
                "Season": s, "Metric": nice,
                "Correlation": spearman(sub[col], sub["total_points"]),
            })
    multi_df = pd.DataFrame(multi_rows)
    fig_multi = px.bar(
        multi_df, x="Metric", y="Correlation", color="Season", barmode="group",
    )
    fig_multi.update_layout(xaxis_tickangle=-30)
    st.plotly_chart(fig_multi, use_container_width=True)

# ------------------------------------------------------------- transfer scout
with tab_scout:
    st.subheader(
        "Hot vs. proven, at any point in the season",
        help="X-axis: season-long average (proven quality). Y-axis: recent form "
        "(current heat). Top-right = premium holds. Top-left = in-form gambles - "
        "classic transfer-in candidates. Bottom-right = proven players in a dip - "
        "classic 'hold or sell?' dilemmas. Bubble size = recent minutes.",
    )

    max_gw = int(features["GW"].max())
    gw = st.slider(
        "As of gameweek", 2, max_gw, min(15, max_gw),
        help="Snapshot the market as it looked going into this gameweek - only "
        "information available at that time is shown.",
    )
    snap = eligible[eligible["GW"] == gw].dropna(subset=["form_points", "season_form_points"])

    if snap.empty:
        st.info("No eligible players for that gameweek.")
    else:
        x_med = snap["season_form_points"].median()
        y_med = snap["form_points"].median()
        fig_scout = px.scatter(
            snap, x="season_form_points", y="form_points",
            color="position", size=snap["form_minutes"].clip(lower=1),
            hover_name="name", hover_data={"team": True, "value": True},
            labels={"season_form_points": "Season-long avg points (proven)",
                    "form_points": "Recent form (hot)"},
        )
        fig_scout.add_hline(y=y_med, line_dash="dot", line_color="gray")
        fig_scout.add_vline(x=x_med, line_dash="dot", line_color="gray")
        st.plotly_chart(fig_scout, use_container_width=True)

    st.subheader(
        "⚠️ Rotation risk: minutes fading",
        help="Players whose recent minutes (4-GW avg) have dropped furthest below "
        "their season average - often the first quantitative sign a player is "
        "losing their starting spot, before the points dry up. Prime "
        "transfer-out candidates.",
    )
    snap2 = eligible[eligible["GW"] == gw].copy()
    season_minutes = (
        features[features["GW"] < gw]
        .groupby("name")["minutes"].mean().rename("season_avg_minutes")
    )
    snap2 = snap2.join(season_minutes, on="name")
    snap2["minutes_delta"] = snap2["form_minutes"] - snap2["season_avg_minutes"]
    fading = (
        snap2[snap2["season_avg_minutes"] >= 45]
        .nsmallest(15, "minutes_delta")
        [["name", "team", "position", "season_avg_minutes", "form_minutes", "minutes_delta", "form_points"]]
    )
    st.dataframe(
        fading.rename(columns={
            "name": "Player", "team": "Team", "position": "Pos",
            "season_avg_minutes": "Season avg mins", "form_minutes": "Recent mins",
            "minutes_delta": "Δ mins", "form_points": "Recent form",
        }).round(1),
        use_container_width=True, hide_index=True,
    )

# ------------------------------------------------------------- value
with tab_value:
    st.subheader(
        "Points per million: who actually justified their price?",
        help="Season total points divided by price. High values = budget enablers "
        "that free up funds elsewhere; premium players rarely top this list but "
        "deliver raw totals no cheap player can.",
    )
    season_totals = (
        eligible.groupby("name", as_index=False)
        .agg(team=("team", "first"), position=("position", "first"),
             price=("value", "last"), total_points=("total_points", "sum"),
             avg_minutes=("minutes", "mean"))
    )
    season_totals = season_totals[season_totals["avg_minutes"] >= 45]
    season_totals["price_m"] = season_totals["price"] / 10
    season_totals["pts_per_m"] = season_totals["total_points"] / season_totals["price_m"]

    fig_val = px.scatter(
        season_totals, x="price_m", y="total_points", color="position",
        hover_name="name", hover_data={"team": True, "pts_per_m": ":.1f"},
        labels={"price_m": "Price (£m)", "total_points": "Season total points"},
    )
    st.plotly_chart(fig_val, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Best value (pts per £m)**")
        st.dataframe(
            season_totals.nlargest(10, "pts_per_m")
            [["name", "team", "position", "price_m", "total_points", "pts_per_m"]]
            .rename(columns={"name": "Player", "team": "Team", "position": "Pos",
                             "price_m": "£m", "total_points": "Pts", "pts_per_m": "Pts/£m"})
            .round(1),
            use_container_width=True, hide_index=True,
        )
    with col_b:
        st.markdown("**Consistency vs. explosiveness**")
        consistency = (
            eligible.groupby("name")
            .agg(mean_pts=("total_points", "mean"), std_pts=("total_points", "std"),
                 position=("position", "first"), games=("total_points", "count"))
            .reset_index()
        )
        consistency = consistency[consistency["games"] >= 10]
        fig_cons = px.scatter(
            consistency, x="mean_pts", y="std_pts", color="position",
            hover_name="name",
            labels={"mean_pts": "Avg points per GW", "std_pts": "Volatility (std dev)"},
        )
        st.plotly_chart(fig_cons, use_container_width=True)
        st.caption(
            "Bottom-right = steady accumulators (safe picks). Top-right = boom-or-bust "
            "haulers (differential/captaincy material)."
        )
