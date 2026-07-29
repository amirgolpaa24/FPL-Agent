"""
FPL Agent - home page of the multipage dashboard.

Run locally:
    pip install streamlit plotly pandas --break-system-packages
    streamlit run dashboard.py

Pages (auto-discovered from the pages/ folder, shown in the sidebar):
    Data Manager      - download raw season data + run preprocessing, in-app
    Captaincy Lab     - backtest & compare captain-picking formulas
    Transfer Signals  - which metrics actually predict points (for transfers & AI)
    Squad Builder     - strategies for picking the initial 15-man squad
    Player Explorer   - per-player form and season leaderboards
    Fixtures          - official FDR and team defensive form
    Distributions     - the raw shape of points / minutes / price data
"""

import pandas as pd
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="FPL Agent", layout="wide", page_icon="⚽")

st.title("⚽ FPL Agent")
st.markdown(
    "An analytics workbench for Fantasy Premier League: explore historical data, "
    "backtest decision rules honestly (no hindsight leakage), and figure out which "
    "signals are worth feeding to an AI assistant for each kind of FPL decision."
)

seasons = lib.list_processed_seasons()

if not seasons:
    st.warning(
        "No data yet. Open the **Data Manager** page (sidebar) to download and "
        "preprocess your first season - everything else lights up after that."
    )
    st.stop()

# --- dataset overview -------------------------------------------------------
st.subheader("Your data at a glance", help="One row per season that's been downloaded and preprocessed. Manage this on the Data Manager page.")

rows = []
for s in seasons:
    df = lib.load_features(s)
    rows.append({
        "Season": s,
        "Gameweeks": int(df["GW"].max()),
        "Players": df["name"].nunique(),
        "Teams": df["team"].nunique(),
        "Rows": len(df),
        "Top scorer": df.groupby("name")["total_points"].sum().idxmax(),
    })
overview = pd.DataFrame(rows)
st.dataframe(overview, use_container_width=True, hide_index=True)

# --- the three FPL decisions -------------------------------------------------
st.subheader("The decisions this tool helps with")

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("#### 🧢 Before the season")
    st.markdown(
        "Pick the initial **15-man squad** under a £100m budget: 2 GK, 5 DEF, "
        "5 MID, 3 FWD, max 3 per club.\n\n"
        "→ **Squad Builder** page: compares selection strategies (proven points, "
        "value-per-million, hybrid) by building squads with only prior-season "
        "knowledge and scoring them against what actually happened."
    )
with c2:
    st.markdown("#### 🔁 Every gameweek: transfers")
    st.markdown(
        "Decide who to **transfer in/out** (1 free transfer, -4 pts each extra).\n\n"
        "→ **Transfer Signals** page: measures which pre-gameweek metrics "
        "(form, ICT, fixtures, price...) actually correlate with future points, "
        "flags fading/rising players, and shows value-for-money - the evidence "
        "base for what an AI should weigh when suggesting transfers."
    )
with c3:
    st.markdown("#### 🧠 Every gameweek: captain & lineup")
    st.markdown(
        "Choose a **captain** (doubled points) and your starting XI.\n\n"
        "→ **Captaincy Lab** page: backtests 7 different captain-picking "
        "formulas across every season you have, so 'which rule works best' is "
        "an empirical answer, not a hunch."
    )

st.divider()

# --- honesty note -------------------------------------------------------------
st.markdown(
    "**Why the backtests can be trusted:** every feature is computed with a time "
    "shift, so a decision for gameweek N only ever sees gameweeks 1 through N-1. "
    "The 'best possible' ceilings shown in comparisons are hindsight - they exist "
    "to calibrate expectations, not because any strategy could reach them."
)
st.caption(
    "Data: vaastav/Fantasy-Premier-League community archive (per-gameweek player stats, "
    "official fixture difficulty ratings). All analysis runs locally - nothing leaves your machine."
)
