"""Backtest and compare captain-picking formulas, per season and across seasons."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import fpl_lib as lib

st.set_page_config(page_title="Captaincy Lab - FPL Agent", layout="wide", page_icon="🧠")

st.title("🧠 Captaincy Lab")
st.markdown(
    "Each formula picks a captain every gameweek using only pre-kickoff information; "
    "we then score it against what actually happened. Captain points count double, "
    "matching real FPL."
)

tab_season, tab_all = st.tabs(["📈 Single-season backtest", "🏆 Cross-season leaderboard"])

# ---------------------------------------------------------------- single season
with tab_season:
    seasons = lib.list_processed_seasons()
    if not seasons:
        st.error("No processed seasons - use the Data Manager page first.")
        st.stop()

    season = st.selectbox(
        "Season", seasons, index=len(seasons) - 1,
        help="Which season to backtest against.",
    )

    default_formulas = [lib.DEFAULT_FORMULA]
    for extra in ("expanding_season_form", "momentum_last_gw"):
        if extra in lib.FORMULA_NAMES and extra not in default_formulas:
            default_formulas.append(extra)

    chosen = st.multiselect(
        "Formulas to compare", lib.FORMULA_NAMES, default=default_formulas,
        format_func=lib.label,
        help="Overlay as many as you like. Hover any formula name in the "
        "cross-season tab for its full definition.",
    )

    if not chosen:
        st.warning("Pick at least one formula.")
    else:
        results_by_formula = {f: lib.compute_backtest(season, f) for f in chosen}
        summaries = {f: lib.summarize(r) for f, r in results_by_formula.items()}
        summaries = {f: s for f, s in summaries.items() if s is not None}

        if not summaries:
            st.info("Not enough history in this season to backtest.")
        else:
            any_summary = next(iter(summaries.values()))

            st.subheader(
                "Season summary",
                help="Gap captured = how much of the distance between a random-ish "
                "eligible pick (floor) and the perfect-hindsight pick (ceiling) the "
                "formula closes. Exact hit rate = share of gameweeks where it picked "
                "the single best captain outright.",
            )
            summary_df = pd.DataFrame([
                {
                    "Formula": lib.label(f),
                    "Total pts (doubled)": s["total_heuristic"],
                    "Avg pts/GW": s["total_heuristic"] / s["n_gw"],
                    "Exact hit rate": s["hit_rate"],
                    "Gap captured": s["gap_capture"],
                }
                for f, s in summaries.items()
            ]).sort_values("Gap captured", ascending=False)
            st.dataframe(
                summary_df.style.format({
                    "Total pts (doubled)": "{:.0f}", "Avg pts/GW": "{:.1f}",
                    "Exact hit rate": "{:.1%}", "Gap captured": "{:.1%}",
                }),
                use_container_width=True, hide_index=True,
            )

            m1, m2 = st.columns(2)
            m1.metric(
                "Ceiling: best possible", f"{any_summary['total_best']:.0f} pts",
                help="What a perfect-hindsight captain would have totaled. Identical "
                "for every formula - it's a property of the season, not the strategy.",
            )
            m2.metric(
                "Floor: field average", f"{any_summary['total_field']:.0f} pts",
                help="What captaining a random regularly-playing player would average.",
            )

            st.subheader(
                "Points per gameweek",
                help="Hover any point to see who was picked that week. Dotted gray "
                "lines are the ceiling/floor references.",
            )
            fig = go.Figure()
            for f, results in results_by_formula.items():
                if results.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=results["GW"], y=results["heuristic_captain_pts_doubled"],
                    mode="lines+markers", name=lib.label(f),
                    hovertext=results["heuristic_captain"],
                ))
            base = next(r for r in results_by_formula.values() if not r.empty)
            fig.add_trace(go.Scatter(
                x=base["GW"], y=base["best_possible_pts_doubled"], mode="lines",
                name="Best possible (hindsight)", line=dict(dash="dot", color="gray"),
                hovertext=base["best_possible_captain"],
            ))
            fig.add_trace(go.Scatter(
                x=base["GW"], y=base["field_average_pts"] * 2, mode="lines",
                name="Field average (x2)", line=dict(dash="dot", color="lightgray"),
            ))
            fig.update_layout(
                xaxis_title="Gameweek", yaxis_title="Points (doubled)",
                hovermode="x unified", legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader(
                "Cumulative points race",
                help="Same data as above, as a running total - easier to see which "
                "formula pulls ahead over a full season.",
            )
            cum_fig = go.Figure()
            for f, results in results_by_formula.items():
                if results.empty:
                    continue
                cum_fig.add_trace(go.Scatter(
                    x=results["GW"], y=results["heuristic_captain_pts_doubled"].cumsum(),
                    mode="lines", name=lib.label(f),
                ))
            cum_fig.add_trace(go.Scatter(
                x=base["GW"], y=base["best_possible_pts_doubled"].cumsum(), mode="lines",
                name="Best possible", line=dict(dash="dot", color="gray"),
            ))
            cum_fig.add_trace(go.Scatter(
                x=base["GW"], y=(base["field_average_pts"] * 2).cumsum(), mode="lines",
                name="Field average", line=dict(dash="dot", color="lightgray"),
            ))
            cum_fig.update_layout(
                xaxis_title="Gameweek", yaxis_title="Cumulative points",
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(cum_fig, use_container_width=True)

            with st.expander("Per-gameweek picks (first selected formula)"):
                st.dataframe(
                    results_by_formula[chosen[0]][[
                        "GW", "heuristic_captain", "heuristic_captain_pts",
                        "best_possible_captain", "best_possible_pts",
                        "field_average_pts", "eligible_pool_size",
                    ]].round(1),
                    use_container_width=True, hide_index=True,
                )

# ---------------------------------------------------------------- cross-season
with tab_all:
    all_seasons = lib.list_processed_seasons()
    lb_seasons = st.multiselect(
        "Seasons to include", all_seasons, default=all_seasons,
        help="More seasons = a more trustworthy ranking. A formula that only wins "
        "in one season may just have been lucky.",
    )

    with st.expander("Formula definitions"):
        for f in lib.FORMULA_NAMES:
            st.markdown(f"**{lib.label(f)}** (`{f}`) — {lib.formula_doc(f)}")

    if not lb_seasons:
        st.warning("Pick at least one season.")
    else:
        rows = []
        with st.spinner("Backtesting every formula across selected seasons..."):
            for s in lb_seasons:
                for f in lib.FORMULA_NAMES:
                    summ = lib.summarize(lib.compute_backtest(s, f))
                    if summ is None:
                        continue
                    rows.append({
                        "season": s, "formula": f,
                        "total_pts": summ["total_heuristic"],
                        "avg_pts_per_gw": summ["total_heuristic"] / summ["n_gw"],
                        "hit_rate": summ["hit_rate"], "gap_capture": summ["gap_capture"],
                    })
        comparison = pd.DataFrame(rows)

        agg = (
            comparison.groupby("formula", as_index=False)
            .agg(seasons_tested=("season", "nunique"),
                 avg_gap_capture=("gap_capture", "mean"),
                 avg_hit_rate=("hit_rate", "mean"),
                 avg_pts_per_gw=("avg_pts_per_gw", "mean"))
            .sort_values("avg_gap_capture", ascending=False)
        )
        agg["Formula"] = agg["formula"].map(lib.label)

        st.subheader(
            "Ranking by average gap-capture",
            help="Gap-capture averages each formula's share of the floor-to-ceiling "
            "distance across the selected seasons - the single best 'how good is "
            "this rule' number we have.",
        )
        fig_bar = px.bar(
            agg, x="avg_gap_capture", y="Formula", orientation="h",
            labels={"avg_gap_capture": "Avg. gap captured"},
            text=agg["avg_gap_capture"].map(lambda v: f"{v:.0%}"),
        )
        fig_bar.update_layout(yaxis=dict(categoryorder="total ascending"), xaxis_tickformat=".0%")
        st.plotly_chart(fig_bar, use_container_width=True)

        best = agg.iloc[0]
        st.success(
            f"**{best['Formula']}** leads: {best['avg_gap_capture']:.1%} average "
            f"gap-capture across {int(best['seasons_tested'])} season(s)."
        )

        st.subheader(
            "Consistency check: formula × season",
            help="A good formula should be green-ish across the whole row. A single "
            "bright cell in an otherwise weak row = one lucky season.",
        )
        heat = comparison.pivot_table(index="formula", columns="season", values="gap_capture")
        heat.index = heat.index.map(lib.label)
        fig_heat = px.imshow(
            heat, aspect="auto", color_continuous_scale="RdYlGn",
            labels=dict(x="Season", y="Formula", color="Gap captured"), text_auto=".0%",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

        with st.expander("Full comparison table"):
            st.dataframe(
                comparison.assign(formula=comparison["formula"].map(lib.label)).rename(columns={
                    "season": "Season", "formula": "Formula", "total_pts": "Total pts",
                    "avg_pts_per_gw": "Avg pts/GW", "hit_rate": "Hit rate",
                    "gap_capture": "Gap captured",
                }).style.format({
                    "Total pts": "{:.0f}", "Avg pts/GW": "{:.1f}",
                    "Hit rate": "{:.1%}", "Gap captured": "{:.1%}",
                }),
                use_container_width=True, hide_index=True,
            )
