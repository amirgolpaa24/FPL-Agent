"""
Interactive front-end for the recommender engine: pick a decision (starting XI,
captain, transfers), choose stats or LLM mode, optionally attach news, and see
the recommendation. This page calls the recommender package directly - the same
code the FastAPI service (api.py) exposes over HTTP - so it works with or
without the API server running.
"""

import os

import pandas as pd
import streamlit as st

import os
from typing import Optional

import fpl_lib as lib
from recommender import data_adapter as da
from recommender import coldstart, engine
from recommender.features import NewsItem, Squad
from recommender.llm import LLMConfig
from recommender.news import DictNewsProvider
from recommender.recommenders import (
    recommend_captain, recommend_lineup, recommend_transfers, recommend_squad,
)
import streamlit.components.v1  # noqa: F401  (used as st.components.v1.html)

try:
    from recommender import live_fpl
    _LIVE_AVAILABLE = os.path.exists(os.path.join(lib.ROOT_DIR, "data", "live", "bootstrap.json"))
except Exception:
    _LIVE_AVAILABLE = False

st.set_page_config(page_title="Recommender - FPL Agent", layout="wide", page_icon="🤖")

st.title("🤖 Recommender")
st.markdown(
    "Turns the backtested models into actual recommendations for the three FPL "
    "decisions. **Stats mode** is deterministic and offline. **LLM mode** feeds the "
    "same numbers plus any news you provide to a language model - the only path "
    "that can react to injuries, rotation and rumours the data can't see."
)

seasons = lib.list_processed_seasons()
if not seasons:
    st.error("No processed seasons - use the Data Manager page first.")
    st.stop()

# --------------------------------------------------------------- sidebar config
with st.sidebar:
    st.header("Data source")
    source = st.radio(
        "Where player data comes from",
        (["live", "historical"] if _LIVE_AVAILABLE else ["historical"]),
        format_func=lambda s: {"live": "🟢 Live current season (FPL API)",
                               "historical": "📚 Historical (backtest/demo)"}[s],
        help="Live uses data/live/ fetched by scripts/fetch_live.py. Historical "
        "uses the processed archive - good for trying the tool before the season starts.",
    )
    if not _LIVE_AVAILABLE:
        st.caption("No live data yet. Run `python3 scripts/fetch_live.py` locally to enable "
                   "the live current-season source.")

    prior_season = None
    season = None
    entry_id = None

    if source == "live":
        # Live: the gameweek is whatever the FPL API says is current/next -
        # no manual picking. Cold-start prior is auto-set to the newest
        # processed season. The only input that matters is your team id.
        _live = live_fpl.LiveData.load()
        gw = _live.current_gameweek()
        prior_season = seasons[-1] if seasons else None

        st.header("Your team")
        entry_id = st.text_input(
            "FPL team ID",
            help="The number in your team's URL (fantasy.premierleague.com/entry/<ID>/...). "
            "Needed so Captain / Starting XI / Transfers run on your ACTUAL squad. "
            "Run `python3 scripts/fetch_live.py --entry <ID>` first to download it. "
            "The Initial Squad tab doesn't need this.",
            placeholder="e.g. 1234567",
        )
        st.caption(f"📅 Detected gameweek: **GW{gw}**"
                   + (f" · cold-start from {prior_season}" if gw <= coldstart.BLEND_FADE_BY_GW else ""))
    else:
        st.header("Scenario")
        season = st.selectbox("Season", seasons, index=len(seasons) - 1)
        max_gw = int(lib.load_features(season)["GW"].max())
        gw = st.slider("Gameweek (decision made going in)", 1, max_gw, min(20, max_gw),
                       help="Only pre-gameweek information is used. GW1 uses cold-start seeding.")
        if gw <= coldstart.BLEND_FADE_BY_GW:
            prior_idx = max(0, seasons.index(season) - 1)
            prior_season = st.selectbox(
                "Cold-start prior season", seasons, index=prior_idx,
                help="GW1-5 have little/no within-season form, so it's seeded from this prior season.",
            )
    formula = st.selectbox(
        "Scoring formula", lib.FORMULA_NAMES,
        index=lib.FORMULA_NAMES.index(engine.DEFAULT_SCORING_FORMULA),
        format_func=lib.label,
        help="The model used to score expected points. Default is the "
        "backtest-winning season-long form model.",
    )

    st.header("Mode")
    mode = st.radio(
        "Decision mode", ["stats", "llm"],
        format_func=lambda m: {"stats": "📊 Stats (deterministic)",
                               "llm": "🤖 LLM (stats + news)"}[m],
        help="LLM mode needs a provider configured below and, ideally, some news.",
    )

    llm_cfg = None
    if mode == "llm":
        st.header("LLM provider")
        # Credentials come from the project .env / environment - no key entered
        # in the UI. Provider & model can still be overridden per session.
        _env_cfg = LLMConfig.from_env()
        provider = st.selectbox(
            "Provider", ["openai", "anthropic", "ollama"],
            index=["openai", "anthropic", "ollama"].index(_env_cfg.provider)
            if _env_cfg.provider in ("openai", "anthropic", "ollama") else 0,
        )
        model = st.text_input(
            "Model", value=_env_cfg.model or "",
            placeholder="leave blank for provider default",
        )
        temperature = st.slider("Temperature", 0.0, 1.0, _env_cfg.temperature, 0.05)
        llm_cfg = LLMConfig.from_env(
            provider=provider, model=model or None, temperature=temperature,
        )
        if llm_cfg.api_key:
            st.caption("🔑 API key loaded from your .env / environment.")
        else:
            st.warning(
                "No API key found. Add `OPENAI_API_KEY=...` (or `LLM_API_KEY`) to a "
                "`.env` file in the project root, or export it in your shell.",
                icon="⚠️",
            )
        st.caption(
            "If the call fails (bad key/model, provider down), the app falls back "
            "to the stats recommendation and tells you."
        )

@st.cache_data(show_spinner=False)
def get_pool(source: str, season: Optional[str], gw: int, prior: Optional[str]) -> list:
    if source == "live":
        return live_fpl.build_pool(gameweek=gw, prior_season=prior)
    players = da.player_pool(season, gw, min_minutes=0 if gw <= 1 else 1)
    if prior:
        players = coldstart.seed_pool(players, prior, gw)
    return players


def get_sample_squad(source: str, season: Optional[str], gw: int, prior: Optional[str],
                     bank: float = 0.0, free_transfers: int = 1) -> Squad:
    """A squad to demo lineup/transfers. Live mode uses your fetched entry if
    present, else a budget-built squad from the (seeded) pool."""
    pool_local = get_pool(source, season, gw, prior)
    # greedy budget-respecting 15 from the pool (works for both sources / GW1)
    ranked = sorted(pool_local, key=lambda p: (p.season_form_points or 0), reverse=True)
    quota = dict(Squad.QUOTA); by_club = {}; chosen = []; names = set(); spent = 0.0

    def min_fill(remaining, exclude):
        tot = 0.0
        for pos, need in remaining.items():
            if need <= 0:
                continue
            pr = sorted(p.price for p in pool_local if p.position == pos and p.name not in exclude)
            if len(pr) < need:
                return float("inf")
            tot += sum(pr[:need])
        return tot

    for p in ranked:
        if quota.get(p.position, 0) <= 0 or p.name in names or by_club.get(p.team, 0) >= 3:
            continue
        rem = {pos: (n - 1 if pos == p.position else n) for pos, n in quota.items()}
        if spent + p.price + min_fill(rem, names | {p.name}) > 100.0 + 1e-6:
            continue
        chosen.append(p); names.add(p.name); spent += p.price
        quota[p.position] -= 1; by_club[p.team] = by_club.get(p.team, 0) + 1
        if all(v == 0 for v in quota.values()):
            break
    return Squad(players=chosen, bank=bank, free_transfers=free_transfers)


def get_user_squad(entry_id: str | None, gw: int, prior: Optional[str]):
    """Load the user's ACTUAL squad from a fetched entry file. Returns
    (squad, message): squad is None when unavailable, with a reason to show."""
    if not entry_id:
        return None, None
    try:
        eid = int(str(entry_id).strip())
    except ValueError:
        return None, "That team ID isn't a number."
    try:
        squad = live_fpl.build_squad_from_entry(eid, gameweek=gw, prior_season=prior)
        return squad, None
    except FileNotFoundError:
        return None, (f"No downloaded data for team {eid}. Run "
                      f"`python3 scripts/fetch_live.py --entry {eid}` first.")
    except Exception as e:
        return None, f"Could not load your squad: {e}"


_season = None if source == "live" else season
user_squad, user_squad_msg = (get_user_squad(entry_id, int(gw), prior_season)
                              if source == "live" else (None, None))

if source == "live":
    try:
        _fresh = live_fpl.LiveData.load().freshness()
        if _fresh["stale"]:
            st.warning(
                "⚠️ The FPL API is still serving the **previous, completed season** "
                f"({_fresh['events_finished']}/{_fresh['events_total']} gameweeks finished, "
                "no upcoming fixtures published yet). The new season's data hasn't rolled "
                "over in the API yet — this usually happens within a week or two of the "
                "opening match. Any recommendation right now is based on **last season's** "
                "fixtures/stats. Re-run `scripts/fetch_live.py` closer to kickoff to get the "
                "real upcoming season.",
                icon="📅",
            )
    except Exception:
        pass

pool = get_pool(source, _season, int(gw), prior_season)

st.caption(
    f"Source: **{source}** · gameweek **{gw}** · pool size **{len(pool)}**"
    + (f" · cold-start seeded from **{prior_season}** "
       f"(prior weight {coldstart.blended_weight(int(gw)):.0%})" if prior_season else "")
)

tab_squad, tab_cap, tab_line, tab_trans = st.tabs(
    ["🧢 Initial Squad", "🧠 Captain", "📋 Starting XI", "🔁 Transfers"]
)

# ------------------------------------------------------------------- news editor
def news_editor(names: list[str], key: str) -> DictNewsProvider:
    """Let the user attach free-text news to specific players (LLM mode only)."""
    with st.expander("📰 Attach news / injury / rumour context (LLM mode)", expanded=False):
        st.caption(
            "This is the external context the stats can't capture. Add a row per "
            "player you have news on - the LLM weighs it against the numbers."
        )
        empty = pd.DataFrame({
            "player": pd.Series([], dtype="object"),
            "category": pd.Series([], dtype="object"),
            "headline": pd.Series([], dtype="object"),
            "sentiment": pd.Series([], dtype="float64"),
        })
        editor = st.data_editor(
            empty,
            num_rows="dynamic", key=key, use_container_width=True,
            column_config={
                "player": st.column_config.SelectboxColumn("Player", options=sorted(names)),
                "category": st.column_config.SelectboxColumn(
                    "Category", options=["injury", "rotation", "transfer", "form", "suspension", "rumour", "general"]),
                "headline": st.column_config.TextColumn("Headline"),
                "sentiment": st.column_config.NumberColumn("Sentiment", min_value=-1.0, max_value=1.0, step=0.1),
            },
        )
        by_player: dict[str, list[NewsItem]] = {}
        for _, r in editor.iterrows():
            if not r.get("player") or not r.get("headline"):
                continue
            by_player.setdefault(r["player"], []).append(NewsItem(
                headline=str(r["headline"]),
                category=str(r.get("category") or "general"),
                sentiment=(float(r["sentiment"]) if pd.notna(r.get("sentiment")) else None),
            ))
        if by_player:
            st.success(f"Attached news for {len(by_player)} player(s).")
        return DictNewsProvider(by_player)


def show_stats_anchor(result: dict):
    if result.get("mode") == "llm" and "stats_anchor" in result:
        with st.expander("What the pure-stats model said (anchor given to the LLM)"):
            st.json(result["stats_anchor"])
    if result.get("llm_failed"):
        st.warning(f"LLM call failed ({result.get('llm_error')}). Showing the "
                   f"stats recommendation instead.", icon="⚠️")


# --------------------------------------------------- FPL-style pitch renderer
_POS_COLORS = {"GK": "#eb1a4a", "DEF": "#00c2b8", "MID": "#04f5ff", "FWD": "#e90052"}
_SHIRT_BASE = "https://fantasy.premierleague.com/dist/img/shirts/standard"


@st.cache_data(show_spinner=False)
def team_code_lookup() -> dict:
    """Map team name -> FPL team code from the live bootstrap (if present),
    so jerseys work for both live and historical squads of current PL teams."""
    path = os.path.join(lib.ROOT_DIR, "data", "live", "bootstrap.json")
    if not os.path.exists(path):
        return {}
    import json
    try:
        teams = json.load(open(path))["teams"]
        return {t["name"]: t["code"] for t in teams}
    except Exception:
        return {}


def _jersey_url(code: int, position: str) -> str:
    # Goalkeepers have a distinct kit ("_1"); outfielders share the base shirt.
    suffix = "_1" if position == "GK" else ""
    return f"{_SHIRT_BASE}/shirt_{code}{suffix}-66.png"


def _display_name(name: str) -> str:
    """'Erling Haaland' -> 'E. Haaland'; 'Bruno Borges Fernandes' -> 'B. Fernandes'."""
    parts = name.split()
    if len(parts) >= 2:
        return f"{parts[0][0]}. {parts[-1]}"
    return name


def _card(p: dict, captain: str | None, vice: str | None, code_map: dict) -> str:
    name = p.get("name", "")
    label = _display_name(name)
    price = p.get("price")
    price_str = f"£{price}m" if price is not None else ""
    xp = p.get("expected_score")
    xp_str = f"{xp:.2f} xP" if isinstance(xp, (int, float)) else ""
    badge = ""
    if name == captain:
        badge = "<span class='cap'>C</span>"
    elif name == vice:
        badge = "<span class='cap vice'>V</span>"

    code = p.get("team_code") or code_map.get(p.get("team"))
    if code:
        icon = (f"<img class='shirt' src='{_jersey_url(code, p.get('position'))}' "
                f"onerror=\"this.style.visibility='hidden'\"/>")
    else:
        color = _POS_COLORS.get(p.get("position"), "#04f5ff")
        icon = f"<div class='shirt' style='background:{color};border-radius:8px;'></div>"

    tooltip = f"<div class='xp'>{xp_str}</div>" if xp_str else ""
    return (
        f"<div class='player'>"
        f"{tooltip}{icon}"
        f"<div class='pname'>{label}{badge}</div>"
        f"<div class='pprice'>{price_str}</div>"
        f"</div>"
    )


_PITCH_CSS = """
<style>
  .player { position:relative; display:inline-flex; flex-direction:column;
            align-items:center; margin:6px 10px; width:96px;
            transition:transform .15s ease; cursor:pointer; }
  .player:hover { transform:scale(1.25); z-index:10; }
  .shirt { width:52px; height:52px; object-fit:contain;
           filter:drop-shadow(0 2px 3px rgba(0,0,0,.45)); }
  .pname { background:#37003c; color:#fff; font-size:12px; font-weight:600;
           padding:2px 7px; border-radius:4px 4px 0 0; margin-top:3px; max-width:96px;
           overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pprice { background:#fff; color:#37003c; font-size:11px; padding:1px 7px;
            border-radius:0 0 4px 4px; }
  .cap { background:#fff; color:#37003c; border-radius:50%; padding:0 5px;
         font-size:10px; font-weight:700; margin-left:4px; }
  .cap.vice { background:#00ff87; }
  .xp { visibility:hidden; opacity:0; position:absolute; bottom:100%; left:50%;
        transform:translateX(-50%); background:#00ff87; color:#37003c;
        font-size:11px; font-weight:700; padding:2px 8px; border-radius:5px;
        white-space:nowrap; transition:opacity .15s ease; margin-bottom:4px;
        box-shadow:0 2px 6px rgba(0,0,0,.4); }
  .player:hover .xp { visibility:visible; opacity:1; }
</style>
"""


def render_pitch(starting_xi: list[dict], bench: list[dict],
                 captain: str | None = None, vice: str | None = None):
    """Render an FPL-style pitch with the XI in position rows + a bench strip."""
    code_map = team_code_lookup()
    rows = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for p in starting_xi:
        rows.get(p.get("position"), rows["MID"]).append(p)

    def row_html(players):
        cards = "".join(_card(p, captain, vice, code_map) for p in players)
        return (f"<div style='display:flex;justify-content:center;align-items:center;"
                f"flex-wrap:wrap;margin:10px 0;'>{cards}</div>")

    pitch_rows = "".join(row_html(rows[pos]) for pos in ["GK", "DEF", "MID", "FWD"] if rows[pos])
    bench_html = "".join(_card(p, captain, vice, code_map) for p in bench)

    html = f"""
    {_PITCH_CSS}
    <div style='font-family:system-ui,sans-serif;'>
      <div style='background:linear-gradient(#2e8b57,#1e6b3c);border-radius:10px;
                  padding:18px 8px;background-image:repeating-linear-gradient(
                  0deg,rgba(255,255,255,.04) 0 40px,transparent 40px 80px);'>
        {pitch_rows}
      </div>
      <div style='margin-top:10px;background:#2a2540;border-radius:10px;padding:10px 8px;'>
        <div style='color:#aaa;font-size:11px;text-align:center;margin-bottom:6px;'>BENCH</div>
        <div style='display:flex;justify-content:center;flex-wrap:wrap;'>{bench_html}</div>
      </div>
    </div>
    """
    st.components.v1.html(html, height=140 + 108 * sum(1 for pos in rows if rows[pos]) + 70)


# --------------------------------------------------------------- initial squad tab
with tab_squad:
    st.subheader("Initial 15-man squad")
    st.caption(
        "The big pre-season decision: pick your starting 15 under £100m (2 GK, 5 DEF, "
        "5 MID, 3 FWD, max 3 per club). Stats mode optimises on form + price + fixtures; "
        "LLM mode also weighs the news you attach."
    )
    budget = st.slider("Budget (£m)", 90.0, 100.0, 100.0, 0.5, key="squad_budget")

    news = None
    if mode == "llm":
        shortlist_names = engine.expected_points(pool, formula).sort_values(
            "expected_score", ascending=False)["name"].head(70).tolist()
        news = news_editor(shortlist_names, "news_squad")

    if st.button("Build squad", type="primary", key="btn_squad"):
        with st.spinner("Assembling squad..."):
            res = recommend_squad(pool, budget=budget, mode=mode, formula=formula,
                                  llm_config=llm_cfg, news_provider=news)
        show_stats_anchor(res)
        if res.get("error"):
            st.error(res["error"])
        else:
            anchor = res.get("stats_anchor", res)  # LLM shares stats layout via anchor
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Cost", f"£{anchor.get('total_cost', res.get('total_cost', 0)):.1f}m")
            c2.metric("Formation", res.get("formation") or anchor.get("formation") or "—")
            c3.metric("Captain", res.get("captain") or anchor.get("captain") or "—")
            c4.metric("Exp. XI pts", f"{anchor.get('expected_xi_points', 0):.1f}")

            if res.get("reasoning"):
                st.markdown(f"**Reasoning:** {res['reasoning']}")
            if res.get("mode") == "llm" and res.get("llm_squad_valid") is False:
                st.warning("The LLM's squad broke a rule "
                           f"({', '.join(res.get('llm_squad_issues', []))}); showing the "
                           "validated stats squad on the pitch instead.", icon="⚠️")

            # Pitch: prefer a valid LLM XI, else the stats anchor's XI.
            xi = anchor.get("best_xi", [])
            bench = anchor.get("bench", [])
            cap = res.get("captain") or anchor.get("captain")
            vice = res.get("vice_captain") or anchor.get("vice_captain")
            if xi and isinstance(xi[0], dict):
                render_pitch(xi, bench, cap, vice)

            with st.expander("Full 15 as a table"):
                st.dataframe(pd.DataFrame(anchor.get("squad", [])),
                             use_container_width=True, hide_index=True)


# ------------------------------------------------------------------- captain tab
with tab_cap:
    st.subheader("Captain recommendation")
    st.caption("Captain's points are doubled - this picks who to armband this gameweek.")

    if user_squad is not None:
        # Real FPL: you can only captain a player in your own squad.
        st.info(f"Choosing from **your squad** (team {entry_id}).", icon="👤")
        candidates = list(user_squad.players)
        cand_names = [p.name for p in candidates]
    else:
        if source == "live" and user_squad_msg:
            st.warning(user_squad_msg + " Meanwhile, choosing from the whole player pool.",
                       icon="ℹ️")
        n_cand = st.slider("Candidate pool size (top by expected score)", 5, 30, 12,
                           help="We pre-rank the gameweek's players and offer the top N as captain candidates.")
        scored = engine.expected_points(pool, formula).sort_values("expected_score", ascending=False)
        cand_names = scored.head(n_cand)["name"].tolist()
        candidates = [p for p in pool if p.name in cand_names]

    news = news_editor(cand_names, "news_cap") if mode == "llm" else None

    if st.button("Recommend captain", type="primary", key="btn_cap"):
        with st.spinner("Thinking..."):
            res = recommend_captain(candidates, mode=mode, formula=formula,
                                    llm_config=llm_cfg, news_provider=news)
        show_stats_anchor(res)
        c1, c2 = st.columns([1, 2])
        c1.metric("Captain", res.get("recommendation") or "—")
        if res.get("vice_captain"):
            c1.metric("Vice", res["vice_captain"])
        if res.get("confidence") is not None:
            c1.metric("Confidence", f"{res['confidence']:.0%}")
        c2.markdown(f"**Reasoning:** {res.get('reasoning', '—')}")
        if res.get("key_factors"):
            c2.markdown("**Key factors:** " + ", ".join(res["key_factors"]))
        if res.get("ranked"):
            st.dataframe(pd.DataFrame(res["ranked"]), use_container_width=True, hide_index=True)


# ------------------------------------------------------------------- lineup tab
with tab_line:
    st.subheader("Starting XI recommendation")
    st.caption("Picks the best 11 + bench + captain from a 15-man squad, over all legal formations.")

    if user_squad is not None:
        st.info(f"Using **your squad** (team {entry_id}).", icon="👤")
        squad = user_squad
    else:
        if source == "live" and user_squad_msg:
            st.warning(user_squad_msg + " Showing a demo squad for now.", icon="ℹ️")
        else:
            st.markdown("Using an auto-built sample squad for this gameweek (a realistic legal £100m squad).")
        squad = get_sample_squad(source, _season, int(gw), prior_season, bank=0.0, free_transfers=1)
    with st.expander("Squad used"):
        st.dataframe(
            pd.DataFrame([{"name": p.name, "pos": p.position, "team": p.team,
                           "price": p.price, "season_form": p.season_form_points}
                          for p in squad.players]),
            use_container_width=True, hide_index=True,
        )

    news = news_editor([p.name for p in squad.players], "news_line") if mode == "llm" else None

    if st.button("Recommend starting XI", type="primary", key="btn_line"):
        with st.spinner("Thinking..."):
            res = recommend_lineup(squad, mode=mode, formula=formula,
                                   llm_config=llm_cfg, news_provider=news)
        show_stats_anchor(res)
        if res.get("error"):
            st.error(res["error"])
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Formation", res.get("formation") or "—")
            c2.metric("Captain", res.get("captain") or "—")
            c3.metric("Vice", res.get("vice_captain") or "—")
            st.markdown(f"**Reasoning:** {res.get('reasoning', '—')}")
            xi = res.get("starting_xi", [])
            if xi and isinstance(xi[0], dict):
                st.markdown("**Starting XI**")
                st.dataframe(pd.DataFrame(xi), use_container_width=True, hide_index=True)
                st.markdown("**Bench**")
                st.dataframe(pd.DataFrame(res.get("bench", [])), use_container_width=True, hide_index=True)
            else:
                st.markdown("**Starting XI:** " + ", ".join(xi))
                if res.get("bench_order"):
                    st.markdown("**Bench:** " + ", ".join(res["bench_order"]))


# ------------------------------------------------------------------- transfers tab
with tab_trans:
    st.subheader("Transfer recommendation")
    st.caption("Suggests swaps that raise expected points, respecting budget, positions and the -4 hit.")

    if user_squad is not None:
        st.info(f"Suggesting transfers for **your squad** (team {entry_id}, "
                f"bank £{user_squad.bank:.1f}m, {user_squad.free_transfers} free transfer(s)).",
                icon="👤")
        squad = user_squad
    else:
        if source == "live" and user_squad_msg:
            st.warning(user_squad_msg + " Showing transfers for a demo squad meanwhile.", icon="ℹ️")
        col1, col2 = st.columns(2)
        bank = col1.number_input("Bank (£m)", 0.0, 20.0, 2.0, 0.5)
        free_t = col2.number_input("Free transfers", 0, 5, 1)
        squad = get_sample_squad(source, _season, int(gw), prior_season,
                                 bank=bank, free_transfers=int(free_t))

    news = news_editor([p.name for p in pool], "news_trans") if mode == "llm" else None

    if st.button("Recommend transfers", type="primary", key="btn_trans"):
        with st.spinner("Thinking..."):
            res = recommend_transfers(squad, pool, mode=mode, formula=formula,
                                      llm_config=llm_cfg, news_provider=news)
        show_stats_anchor(res)
        st.markdown(f"**Reasoning:** {res.get('reasoning', '—')}")
        if res.get("transfers"):  # LLM shape
            st.dataframe(pd.DataFrame(res["transfers"]), use_container_width=True, hide_index=True)
            if res.get("take_hit") is not None:
                st.caption(f"Take a points hit: {res['take_hit']}")
        if res.get("suggestions"):  # stats shape
            st.dataframe(pd.DataFrame(res["suggestions"]), use_container_width=True, hide_index=True)
