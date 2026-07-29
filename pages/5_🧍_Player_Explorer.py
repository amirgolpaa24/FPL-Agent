"""Per-player deep dive: form curves, leaderboards, head-to-head comparison."""

import pandas as pd
import plotly.express as px
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="Player Explorer - FPL Agent", layout="wide", page_icon="🧍")

st.title("🧍 Player Explorer")

season = lib.season_picker()
if season is None:
    st.stop()

features = lib.load_features(season)
positions = sorted(features["position"].dropna().unique())
teams = sorted(features["team"].dropna().unique())

with st.sidebar.expander("🔎 Filters", expanded=True):
    selected_positions = st.multiselect("Position", positions, default=positions)
    selected_teams = st.multiselect("Team", teams, default=teams)
    min_minutes = st.slider(
        "Min avg minutes (form window)", 0, 90, 0, step=5,
        help="Hides fringe players who weren't getting regular game time.",
    )

filtered = features[
    features["position"].isin(selected_positions)
    & features["team"].isin(selected_teams)
    & (features["form_minutes"].fillna(0) >= min_minutes)
]

player_totals = (
    filtered.groupby("name", as_index=False)
    .agg(team=("team", "first"), position=("position", "first"),
         total_points=("total_points", "sum"), avg_points=("total_points", "mean"),
         avg_minutes=("minutes", "mean"), price=("value", "last"))
    .sort_values("total_points", ascending=False)
)

st.subheader(
    "Compare players",
    help="Left: what each player actually scored, gameweek by gameweek. Right: the "
    "rolling form number every formula sees *going into* each gameweek. The lag "
    "between the two is the fundamental difficulty of FPL prediction.",
)
chosen = st.multiselect(
    "Players", player_totals["name"].tolist(),
    default=player_totals["name"].head(5).tolist(),
)

if chosen:
    subset = filtered[filtered["name"].isin(chosen)].sort_values("GW")
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.line(
            subset, x="GW", y="total_points", color="name", markers=True,
            labels={"total_points": "Actual points", "name": "Player"},
            title="Actual points per gameweek",
        )
        st.plotly_chart(fig1, use_container_width=True)
    with col_b:
        fig2 = px.line(
            subset, x="GW", y="form_points", color="name", markers=True,
            labels={"form_points": "Rolling form", "name": "Player"},
            title="Form going into each gameweek",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader(
        "Head-to-head profile",
        help="Season aggregates for the selected players, side by side.",
    )
    h2h = player_totals[player_totals["name"].isin(chosen)].copy()
    h2h["price_m"] = h2h["price"] / 10
    h2h["pts_per_m"] = h2h["total_points"] / h2h["price_m"]
    st.dataframe(
        h2h[["name", "team", "position", "total_points", "avg_points",
             "avg_minutes", "price_m", "pts_per_m"]]
        .rename(columns={
            "name": "Player", "team": "Team", "position": "Pos",
            "total_points": "Total pts", "avg_points": "Avg pts/GW",
            "avg_minutes": "Avg mins", "price_m": "£m (end)", "pts_per_m": "Pts/£m",
        }).round(1),
        use_container_width=True, hide_index=True,
    )
else:
    st.info("Pick one or more players above.")

st.divider()
st.subheader(
    "Season leaderboard",
    help="All players passing the sidebar filters, ranked by season total points.",
)
st.dataframe(
    player_totals.round(1), use_container_width=True, hide_index=True, height=400,
)
