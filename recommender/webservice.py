"""
High-level service layer for the web frontend. Unlike the low-level
/recommend/* API (which takes explicit player lists), these functions build the
player pool / squad SERVER-SIDE from live or historical data, run the
recommender, and return browser-ready JSON. The single-page app calls these.

Kept framework-free (no FastAPI here) so it's unit-testable on its own.
"""

from __future__ import annotations

import os
from typing import Optional

from . import coldstart, engine
from .features import Squad
from .llm import LLMConfig
from .news import DictNewsProvider, coerce_news
from .recommenders import (
    recommend_squad, recommend_captain, recommend_lineup, recommend_transfers,
)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROCESSED = os.path.join(_ROOT, "data", "processed")
_LIVE = os.path.join(_ROOT, "data", "live")


def _live_available() -> bool:
    return os.path.exists(os.path.join(_LIVE, "bootstrap.json"))


def _processed_seasons() -> list[str]:
    if not os.path.isdir(_PROCESSED):
        return []
    return sorted(
        d for d in os.listdir(_PROCESSED)
        if os.path.exists(os.path.join(_PROCESSED, d, "player_gw_features.csv"))
        and d != "test-season"
    )


def status() -> dict:
    """Everything the frontend needs to render its initial state."""
    out = {
        "live_available": _live_available(),
        "seasons": _processed_seasons(),
        "formulas": list(engine.FORMULAS.keys()),
        "default_formula": engine.DEFAULT_SCORING_FORMULA,
        "live_default_formula": "fpl_official",
        "score_label": "xP",
        "score_desc": ("On live data this is FPL's own official expected points "
                       "for the upcoming gameweek (their ep_next figure). On "
                       "historical data it's our composite model score."),
        "gameweek": None,
        "stale": None,
        "freshness": None,
    }
    cfg = LLMConfig.from_env()
    out["llm_configured"] = bool(cfg.api_key)
    out["llm_provider"] = cfg.provider
    out["llm_model"] = cfg.model or "(provider default)"

    try:
        import database as db
        out["database"] = db.stats()
    except Exception:
        out["database"] = {"exists": False}

    if out["live_available"]:
        from . import live_fpl
        live = live_fpl.LiveData.load()
        out["gameweek"] = live.current_gameweek()
        fresh = live.freshness()
        out["stale"] = fresh["stale"]
        out["freshness"] = fresh
    return out


def _build_pool(source: str, season: Optional[str], gw: int, prior: Optional[str]):
    if source == "live":
        from . import live_fpl
        return live_fpl.build_pool(gameweek=gw, prior_season=prior)
    players = _data_pool(season, gw)
    if prior:
        players = coldstart.seed_pool(players, prior, gw)
    return players


def _data_pool(season: str, gw: int):
    from . import data_adapter
    return data_adapter.player_pool(season, gw, min_minutes=0 if gw <= 1 else 1)


def _user_squad(entry_id, gw: int, prior: Optional[str]):
    from . import live_fpl
    return live_fpl.build_squad_from_entry(int(entry_id), gameweek=gw, prior_season=prior)


def _demo_squad(pool, bank: float, free_transfers: int, budget: float = 100.0) -> Squad:
    """Budget-legal 15 from the pool, for when no real squad is supplied."""
    sq = engine.stats_squad(pool, budget=budget)
    if "error" in sq:
        raise ValueError(sq["error"])
    names = {p["name"] for p in sq["squad"]}
    players = [p for p in pool if p.name in names]
    return Squad(players=players, bank=bank, free_transfers=free_transfers)


def _context(payload: dict) -> dict:
    """Build everything a decision needs (pool, squad, news, candidates) from a
    request payload. Deterministic given the same data files, so prepare() and
    finalize() reconstruct identical context across two HTTP calls (BYOK flow).
    Returns {'error': ...} on failure."""
    decision = payload.get("decision", "squad")
    mode = payload.get("mode", "stats")
    source = payload.get("source") or ("live" if _live_available() else "historical")
    # Live: default to FPL's OWN official expected points (ep_next). Historical
    # backtests have no such field, so fall back to our composite model.
    default_formula = "fpl_official" if source == "live" else "composite"
    formula = payload.get("formula") or default_formula
    if formula not in engine.FORMULAS:
        return {"error": f"unknown formula {formula!r}"}

    seasons = _processed_seasons()
    prior = None
    if source == "live":
        from . import live_fpl
        live = live_fpl.LiveData.load()
        gw = int(payload.get("gameweek") or live.current_gameweek())
        prior = seasons[-1] if seasons else None
        season = None
    else:
        season = payload.get("season") or (seasons[-1] if seasons else None)
        if not season:
            return {"error": "no processed seasons available"}
        gw = int(payload.get("gameweek") or 1)
        if gw <= coldstart.BLEND_FADE_BY_GW:
            idx = seasons.index(season) if season in seasons else len(seasons) - 1
            prior = seasons[max(0, idx - 1)]

    try:
        pool = _build_pool(source, season, gw, prior)
    except Exception as e:
        return {"error": f"could not load player data: {e}"}
    if not pool:
        return {"error": "player pool is empty for this gameweek"}

    # news: gathered automatically in AI mode, weighed by source reliability.
    news_provider = None
    news_used = []
    if mode == "llm":
        from . import news_aggregator
        try:
            by_name = news_aggregator.gather(pool, use_rss=payload.get("crawl_news", True))
        except Exception:
            by_name = {}
        for item in payload.get("news", []) or []:
            pname = item.get("player")
            if pname:
                by_name.setdefault(pname, []).extend(coerce_news([item]))
        if by_name:
            news_provider = DictNewsProvider(by_name)
            news_used = news_aggregator.flatten(by_name)

    # squad for decisions that need one
    entry_id = payload.get("entry_id")
    squad = None
    squad_note = None
    if decision in ("lineup", "transfers", "captain") and source == "live" and entry_id:
        try:
            squad = _user_squad(entry_id, gw, prior)
        except FileNotFoundError:
            squad_note = (f"No downloaded data for team {entry_id}. Run "
                          f"scripts/fetch_live.py --entry {entry_id} first.")
        except ValueError:
            squad_note = ("Your team isn't available from the FPL API yet - it only "
                          "publishes your picks after the gameweek deadline.")
        except Exception as e:
            squad_note = f"Could not load your squad: {e}"

    # captain candidate pool (deterministic ordering)
    candidates = None
    if decision == "captain":
        if squad is not None:
            candidates = list(squad.players)
        else:
            scored = engine.expected_points(pool, formula).sort_values(
                "expected_score", ascending=False)
            names = set(scored.head(int(payload.get("candidates", 15)))["name"])
            candidates = [p for p in pool if p.name in names]
    elif decision in ("lineup", "transfers") and squad is None:
        squad = _demo_squad(pool, float(payload.get("bank", 2.0)),
                            int(payload.get("free_transfers", 1)))

    return {
        "decision": decision, "mode": mode, "source": source, "formula": formula,
        "season": season, "gw": gw, "prior": prior, "pool": pool,
        "news_provider": news_provider, "news_used": news_used,
        "squad": squad, "squad_note": squad_note, "candidates": candidates,
        "entry_id": entry_id,
    }


def _meta_for(ctx: dict) -> dict:
    return {
        "decision": ctx["decision"], "mode": ctx["mode"], "source": ctx["source"],
        "gameweek": ctx["gw"], "season": ctx["season"], "prior_season": ctx["prior"],
        "pool_size": len(ctx["pool"]), "formula": ctx["formula"],
        "used_real_squad": ctx["squad"] is not None and bool(ctx["entry_id"]),
        "squad_note": ctx["squad_note"],
        "news_used": ctx["news_used"][:40],
        "news_count": len(ctx["news_used"]),
    }


def recommend(payload: dict) -> dict:
    """Single entry point for every decision (server-side LLM key, if any).

    payload keys:
      decision: 'squad' | 'captain' | 'lineup' | 'transfers'
      mode:     'stats' | 'llm'
      source:   'live' | 'historical'
      season:   (historical only)
      gameweek: int (optional; live auto-detects)
      budget:   float (squad; default 100)
      entry_id: str/int (live; use the real squad)
      bank, free_transfers: (transfers demo fallback)
      news:     list of {player, headline, category, sentiment}
      formula:  scoring formula name (optional)
    """
    ctx = _context(payload)
    if "error" in ctx:
        return ctx
    decision, mode, formula = ctx["decision"], ctx["mode"], ctx["formula"]
    pool, squad = ctx["pool"], ctx["squad"]

    llm_config = LLMConfig.from_env() if mode == "llm" else None
    common = dict(mode=mode, formula=formula, llm_config=llm_config,
                  news_provider=ctx["news_provider"])

    try:
        if decision == "squad":
            result = recommend_squad(pool, budget=float(payload.get("budget", 100.0)), **common)
        elif decision == "captain":
            result = recommend_captain(ctx["candidates"], **common)
        elif decision == "lineup":
            result = recommend_lineup(squad, **common)
        elif decision == "transfers":
            result = recommend_transfers(squad, pool, **common)
        else:
            return {"error": f"unknown decision {decision!r}"}
    except Exception as e:
        return {"error": f"recommendation failed: {e}"}

    result["_meta"] = _meta_for(ctx)
    _persist(payload, result, ctx)
    return result


def _persist(payload: dict, result: dict, ctx: dict) -> None:
    """Best-effort DB logging - never breaks the response."""
    try:
        import database as db
        db.log_recommendation(payload, result)
        if ctx["news_used"]:
            db.save_news(ctx["news_used"], ctx["season"], ctx["gw"])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# BYOK (bring-your-own-key) flow: the user's LLM key lives ONLY on their
# device. The browser asks us to PREPARE the prompt, calls the LLM provider
# DIRECTLY itself, then posts the raw reply back for us to FINALIZE into a
# validated recommendation. The key never touches this server.
# ---------------------------------------------------------------------------

def prepare(payload: dict) -> dict:
    """Build the exact prompt for a decision so the client can run the LLM
    call itself. Returns {system, prompt, meta}."""
    from . import prompts
    from .news import NullNewsProvider, attach_news
    from .recommenders import _shortlist

    payload = dict(payload, mode="llm")
    ctx = _context(payload)
    if "error" in ctx:
        return ctx
    decision, formula, pool = ctx["decision"], ctx["formula"], ctx["pool"]
    provider = ctx["news_provider"] or NullNewsProvider()

    try:
        if decision == "squad":
            shortlist = _shortlist(pool, formula)
            attach_news(shortlist, provider)
            user = prompts.squad_prompt(shortlist, float(payload.get("budget", 100.0)))
        elif decision == "captain":
            attach_news(ctx["candidates"], provider)
            user = prompts.captain_prompt(ctx["candidates"])
        elif decision == "lineup":
            attach_news(ctx["squad"].players, provider)
            user = prompts.lineup_prompt(ctx["squad"].players)
        elif decision == "transfers":
            attach_news(ctx["squad"].players, provider)
            attach_news(pool, provider)
            user = prompts.transfers_prompt(ctx["squad"].players, pool,
                                            ctx["squad"].bank, ctx["squad"].free_transfers)
        else:
            return {"error": f"unknown decision {decision!r}"}
    except Exception as e:
        return {"error": f"prompt preparation failed: {e}"}

    return {"system": prompts.SYSTEM, "prompt": user, "meta": _meta_for(ctx)}


def finalize(payload: dict, llm_text: str, model: str | None = None) -> dict:
    """Take the raw LLM reply the client obtained with its own key and turn it
    into a validated recommendation (same post-processing as server-side mode).
    Falls back to the stats result if the reply can't be parsed."""
    from .llm import extract_json
    from .recommenders import (
        finalize_captain, finalize_lineup, finalize_squad, finalize_transfers,
    )

    payload = dict(payload, mode="llm")
    ctx = _context(payload)
    if "error" in ctx:
        return ctx
    decision, formula, pool = ctx["decision"], ctx["formula"], ctx["pool"]

    try:
        parsed = extract_json(llm_text or "")
        if not isinstance(parsed, dict):
            raise ValueError("not a JSON object")
    except Exception as e:
        # graceful fallback to stats, same contract as server-side LLM failure
        stats_payload = dict(payload, mode="stats")
        stats_result = recommend(stats_payload)
        stats_result["mode"] = "llm"
        stats_result["llm_failed"] = True
        stats_result["llm_error"] = f"could not parse model reply: {e}"
        stats_result["fallback"] = "stats"
        return stats_result

    try:
        if decision == "squad":
            result = finalize_squad(pool, parsed, model,
                                    budget=float(payload.get("budget", 100.0)),
                                    formula=formula, llm_config=None)
        elif decision == "captain":
            result = finalize_captain(ctx["candidates"], parsed, model, formula)
        elif decision == "lineup":
            result = finalize_lineup(ctx["squad"], parsed, model, formula)
        elif decision == "transfers":
            result = finalize_transfers(ctx["squad"], pool, parsed, model, formula)
        else:
            return {"error": f"unknown decision {decision!r}"}
    except Exception as e:
        return {"error": f"finalization failed: {e}"}

    result["_meta"] = _meta_for(ctx)
    _persist(payload, result, ctx)
    return result
