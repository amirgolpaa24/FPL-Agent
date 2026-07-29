"""The raw statistical shape of the data behind every other page."""

import plotly.express as px
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="Distributions - FPL Agent", layout="wide", page_icon="📊")

st.title("📊 Distributions")
st.markdown(
    "Sanity-check views of the underlying data. Knowing these shapes explains a lot "
    "of *why* the other pages look the way they do."
)

season = lib.season_picker()
if season is None:
    st.stop()

features = lib.load_features(season)

col1, col2 = st.columns(2)
with col1:
    st.subheader(
        "Points per player-gameweek",
        help="Heavily skewed: most appearances score 0-2 points, double-digit hauls "
        "are rare events. This is why captaincy exact-hit rates are naturally low "
        "for every strategy - you're trying to predict outliers.",
    )
    fig7 = px.histogram(features, x="total_points", nbins=30)
    st.plotly_chart(fig7, use_container_width=True)
with col2:
    st.subheader(
        "Minutes per player-gameweek",
        help="Bimodal: players mostly either don't play or play the full 90. There's "
        "very little middle ground, which is why the eligibility filter (min recent "
        "minutes) is so effective at cleaning the candidate pool.",
    )
    fig8 = px.histogram(features, x="minutes", nbins=30)
    st.plotly_chart(fig8, use_container_width=True)

col3, col4 = st.columns(2)
with col3:
    st.subheader(
        "Points by position",
        help="Box plots of per-gameweek points among players who actually played "
        "(>0 minutes). Shows why midfielders/forwards dominate captaincy: higher "
        "ceilings, similar floors.",
    )
    played = features[features["minutes"] > 0]
    fig_pos = px.box(
        played, x="position", y="total_points",
        labels={"position": "", "total_points": "Points per GW"},
        category_orders={"position": ["GK", "DEF", "MID", "FWD"]},
    )
    st.plotly_chart(fig_pos, use_container_width=True)
with col4:
    st.subheader(
        "Price distribution",
        help="Where the market concentrates. The long right tail is the handful of "
        "premium players every budget decision revolves around.",
    )
    latest_prices = features.sort_values("GW").groupby("name")["value"].last() / 10
    fig_price = px.histogram(latest_prices, nbins=30, labels={"value": "Price (£m)"})
    fig_price.update_layout(showlegend=False, xaxis_title="Price (£m)")
    st.plotly_chart(fig_price, use_container_width=True)

st.subheader(
    "Price vs. season total points",
    help="Each dot is a player. Broadly linear - you get what you pay for - but the "
    "vertical spread at each price point is the whole game: finding the dots that "
    "sit far above their price band.",
)
season_totals = (
    features.groupby("name", as_index=False)
    .agg(team=("team", "first"), position=("position", "first"),
         price=("value", "last"), total_points=("total_points", "sum"))
)
season_totals["price_m"] = season_totals["price"] / 10
fig9 = px.scatter(
    season_totals, x="price_m", y="total_points", color="position",
    hover_name="name", hover_data=["team"],
    labels={"price_m": "Price (£m)", "total_points": "Season total points"},
)
st.plotly_chart(fig9, use_container_width=True)
