"""Fixture difficulty and team form - the context around every decision."""

import plotly.express as px
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="Fixtures - FPL Agent", layout="wide", page_icon="🛡️")

st.title("🛡️ Fixtures & Team Form")

season = lib.season_picker()
if season is None:
    st.stop()

features = lib.load_features(season)

if "fixture_difficulty" in features.columns and features["fixture_difficulty"].notna().any():
    st.subheader(
        "Official FPL fixture difficulty",
        help="FPL's own 1-5 rating, published before each gameweek (1 = easiest, "
        "5 = hardest). Green cells = easy fixtures. This is the signal the "
        "baseline captaincy formula uses, and a key input for planning transfer "
        "timing around good fixture runs.",
    )
    fdr_pivot = (
        features.dropna(subset=["fixture_difficulty"])
        .drop_duplicates(subset=["team", "GW"])
        .pivot_table(index="team", columns="GW", values="fixture_difficulty", aggfunc="mean")
    )
    fig_fdr = px.imshow(
        fdr_pivot, aspect="auto", color_continuous_scale="RdYlGn_r", zmin=1, zmax=5,
        labels=dict(x="Gameweek", y="Team", color="Difficulty"),
    )
    st.plotly_chart(fig_fdr, use_container_width=True)

    st.subheader(
        "Easiest fixture runs",
        help="Teams ranked by average official difficulty over a window you choose - "
        "the classic 'whose players should I bring in for the next stretch' view.",
    )
    max_gw = int(features["GW"].max())
    gw_start, gw_end = st.slider(
        "Gameweek window", 1, max_gw, (1, min(6, max_gw)),
        help="Average difficulty is computed over this range of gameweeks.",
    )
    window_fdr = (
        features.dropna(subset=["fixture_difficulty"])
        .drop_duplicates(subset=["team", "GW"])
        .query("@gw_start <= GW <= @gw_end")
        .groupby("team", as_index=False)["fixture_difficulty"].mean()
        .sort_values("fixture_difficulty")
    )
    fig_run = px.bar(
        window_fdr, x="fixture_difficulty", y="team", orientation="h",
        labels={"fixture_difficulty": "Avg difficulty", "team": ""},
        color="fixture_difficulty", color_continuous_scale="RdYlGn_r",
        range_color=[1, 5],
    )
    fig_run.update_layout(yaxis=dict(categoryorder="total descending"), coloraxis_showscale=False)
    st.plotly_chart(fig_run, use_container_width=True)
    st.divider()

st.subheader(
    "Team defensive form over the season",
    help="Each team's rolling goals-conceded average (the last few gameweeks, "
    "time-shifted). Rising line = defense getting leakier = their upcoming "
    "opponents' attackers become more attractive picks.",
)
fig5 = px.line(
    features.dropna(subset=["opp_goals_conceded_form"]).drop_duplicates(["team", "GW"]),
    x="GW", y="opp_goals_conceded_form", color="team",
    labels={"opp_goals_conceded_form": "Rolling goals conceded"},
)
st.plotly_chart(fig5, use_container_width=True)

with st.expander("Proxy vs. official: does recent defensive form agree with FDR?"):
    st.markdown(
        "The heatmap below shows the goals-conceded **proxy** (what we'd use if the "
        "official rating didn't exist). Comparing it with the official FDR heatmap "
        "above shows where the two disagree - and disagreements are interesting: "
        "FDR is set from long-run team strength, so a 'hard' fixture against a team "
        "in terrible defensive form may actually be a good attacking opportunity. "
        "That nuance is exactly the kind of thing an AI layer could exploit."
    )
    pivot = (
        features.dropna(subset=["opp_goals_conceded_form"])
        .drop_duplicates(subset=["team", "opponent_team", "GW"])
        .pivot_table(index="team", columns="GW", values="opp_goals_conceded_form", aggfunc="mean")
    )
    fig6 = px.imshow(
        pivot, aspect="auto", color_continuous_scale="RdYlGn",
        labels=dict(x="Gameweek", y="Team", color="Opp. conceded form"),
    )
    st.plotly_chart(fig6, use_container_width=True)
