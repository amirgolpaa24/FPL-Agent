"""Download raw season data and run preprocessing without leaving the app."""

import os

import pandas as pd
import streamlit as st

import fpl_lib as lib
from fetch_data import fetch_season, KNOWN_SEASONS
from build_dataset import build_features

st.set_page_config(page_title="Data Manager - FPL Agent", layout="wide", page_icon="📥")

st.title("📥 Data Manager")
st.markdown(
    "Download any season's raw data and preprocess it into the feature table every "
    "other page uses - no terminal needed."
)

raw_seasons = lib.list_raw_seasons()
processed_seasons = lib.list_processed_seasons(exclude_synthetic=False)

# --- status table -------------------------------------------------------------
st.subheader(
    "Season status",
    help="Raw = downloaded CSVs from the community archive. Processed = the "
    "lookahead-safe feature table built from them. A season needs both ✅ "
    "to appear in the rest of the app.",
)

status_rows = []
for s in KNOWN_SEASONS:
    row = {"Season": s, "Raw downloaded": "✅" if s in raw_seasons else "—",
           "Processed": "✅" if s in processed_seasons else "—"}
    if s in processed_seasons:
        df = lib.load_features(s)
        row["Rows"] = f"{len(df):,}"
        row["Gameweeks"] = int(df["GW"].max())
    status_rows.append(row)
st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

st.divider()

col_dl, col_proc = st.columns(2)

# --- download -----------------------------------------------------------------
with col_dl:
    st.subheader(
        "1 · Download raw data",
        help="Fetches merged_gw.csv (per-player per-gameweek stats), teams.csv, "
        "fixtures.csv (incl. official difficulty ratings), and players_raw.csv "
        "from the vaastav/Fantasy-Premier-League GitHub archive. Needs internet.",
    )
    dl_choices = st.multiselect(
        "Seasons to download", KNOWN_SEASONS,
        default=[s for s in KNOWN_SEASONS[-1:] if s not in raw_seasons],
        help="Already-downloaded seasons are safe to re-download; files just get refreshed.",
    )
    if st.button("⬇️ Download selected", disabled=not dl_choices, type="primary"):
        progress = st.progress(0.0)
        for i, s in enumerate(dl_choices):
            with st.status(f"Downloading {s}...", expanded=False) as status:
                try:
                    fetch_season(s, lib.RAW_DIR)
                    status.update(label=f"{s} downloaded ✅", state="complete")
                except Exception as e:
                    status.update(label=f"{s} failed: {e}", state="error")
            progress.progress((i + 1) / len(dl_choices))
        st.rerun()

# --- preprocess -----------------------------------------------------------------
with col_proc:
    st.subheader(
        "2 · Preprocess into features",
        help="Builds the per-player per-gameweek feature table: rolling form, "
        "opponent form, official FDR - all time-shifted so gameweek N features "
        "only use information from before gameweek N (no hindsight leakage).",
    )
    proc_choices = st.multiselect(
        "Seasons to preprocess", raw_seasons,
        default=[s for s in raw_seasons if s not in processed_seasons],
        help="Only downloaded seasons appear here. Reprocessing an existing season "
        "overwrites its feature table and automatically refreshes all pages.",
    )
    window = st.slider(
        "Rolling form window (gameweeks)", 2, 10, 4,
        help="How many recent gameweeks the 'recent form' features average over. "
        "4 matches FPL's own form definition; larger = smoother but slower to react.",
    )
    if st.button("⚙️ Preprocess selected", disabled=not proc_choices, type="primary"):
        progress = st.progress(0.0)
        for i, s in enumerate(proc_choices):
            with st.status(f"Processing {s}...", expanded=False) as status:
                try:
                    out_path = build_features(s, window=window, raw_dir=lib.RAW_DIR, out_dir=lib.DATA_DIR)
                    n = len(pd.read_csv(out_path))
                    status.update(label=f"{s} processed ✅ ({n:,} rows)", state="complete")
                except Exception as e:
                    status.update(label=f"{s} failed: {e}", state="error")
            progress.progress((i + 1) / len(proc_choices))
        st.rerun()

st.divider()
st.caption(
    "Data source: github.com/vaastav/Fantasy-Premier-League - a community-maintained "
    "archive of every FPL season. If the newest completed season is missing from the "
    "list, it may not be archived there yet."
)
