"""
High-level orchestration: the public entry points the API and UI call. Each
of the three decisions supports mode="stats" (deterministic, offline) or
mode="llm" (stats + news, judged by a language model).

The LLM path always computes the stats result first and hands it to the model
as an anchor, and always falls back to the stats result (flagged) if the LLM
call or its JSON parsing fails - so a missing key or a flaky provider degrades
gracefully instead of erroring out the whole request.
"""

from __future__ import annotations

from typing import Optional

from . import engine
from .features import PlayerFeatures, Squad
from .llm import LLMClient, LLMConfig, LLMError
from . import prompts
from .news import NewsProvider, NullNewsProvider, attach_news


def _run_llm(system: str, user: str, llm_config: Optional[LLMConfig]) -> tuple[Optional[dict], Optional[str], Optional[str]]:
    """Returns (parsed_json, model_name, error). Never raises - ANY failure
    (bad key, network error, timeout, malformed JSON) degrades to the stats
    result rather than breaking the request."""
    try:
        client = LLMClient(llm_config)
        parsed = client.complete_json(system, user)
        if not isinstance(parsed, dict):
            return None, None, "LLM did not return a JSON object"
        return parsed, client.model, None
    except Exception as e:  # noqa: BLE001 - deliberate catch-all for graceful fallback
        return None, None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# captain
# ---------------------------------------------------------------------------

def recommend_captain(
    candidates: list[PlayerFeatures],
    mode: str = "stats",
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    llm_config: Optional[LLMConfig] = None,
    news_provider: Optional[NewsProvider] = None,
) -> dict:
    stats = engine.stats_captain(candidates, formula=formula)
    if mode == "stats":
        return {"mode": "stats", **stats}

    attach_news(candidates, news_provider or NullNewsProvider())
    user = prompts.captain_prompt(candidates)
    parsed, model, err = _run_llm(prompts.SYSTEM, user, llm_config)

    if parsed is None:
        return {"mode": "llm", "llm_failed": True, "llm_error": err,
                "fallback": "stats", **stats}
    return finalize_captain(candidates, parsed, model, formula, stats=stats)


def finalize_captain(
    candidates: list[PlayerFeatures],
    parsed: dict,
    model: Optional[str],
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    stats: Optional[dict] = None,
) -> dict:
    """Turn an LLM's parsed captain JSON into the final result. Public so the
    BYOK path (browser calls the LLM directly) can post the JSON back."""
    stats = stats or engine.stats_captain(candidates, formula=formula)
    return {
        "mode": "llm",
        "model": model,
        "recommendation": parsed.get("captain"),
        "vice_captain": parsed.get("vice_captain"),
        "confidence": parsed.get("confidence"),
        "overrode_stats": parsed.get("overrode_stats"),
        "key_factors": parsed.get("key_factors", []),
        "reasoning": parsed.get("reasoning"),
        "stats_anchor": stats,
    }


# ---------------------------------------------------------------------------
# lineup
# ---------------------------------------------------------------------------

def recommend_lineup(
    squad: Squad,
    mode: str = "stats",
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    llm_config: Optional[LLMConfig] = None,
    news_provider: Optional[NewsProvider] = None,
) -> dict:
    stats = engine.stats_lineup(squad, formula=formula)
    if mode == "stats":
        return {"mode": "stats", **stats}

    attach_news(squad.players, news_provider or NullNewsProvider())
    user = prompts.lineup_prompt(squad.players)
    parsed, model, err = _run_llm(prompts.SYSTEM, user, llm_config)

    if parsed is None:
        return {"mode": "llm", "llm_failed": True, "llm_error": err,
                "fallback": "stats", **stats}
    return finalize_lineup(squad, parsed, model, formula, stats=stats)


def finalize_lineup(
    squad: Squad,
    parsed: dict,
    model: Optional[str],
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    stats: Optional[dict] = None,
) -> dict:
    """Turn an LLM's parsed lineup JSON into the final result (BYOK-callable).
    Resolves the AI's XI names to real squad players so the pitch renders
    player cards, falling back to the stats XI when names don't resolve."""
    stats = stats or engine.stats_lineup(squad, formula=formula)

    out = {
        "mode": "llm",
        "model": model,
        "formation": parsed.get("formation"),
        "starting_xi": parsed.get("starting_xi", []),
        "bench_order": parsed.get("bench_order", []),
        "captain": parsed.get("captain"),
        "vice_captain": parsed.get("vice_captain"),
        "overrode_stats": parsed.get("overrode_stats"),
        "reasoning": parsed.get("reasoning"),
        "stats_anchor": stats,
    }

    # Render the AI's XI as real player cards when its names resolve to a
    # legal formation; otherwise the frontend falls back to the stats anchor.
    xi_players = _resolve_names(parsed.get("starting_xi", []) or [], squad.players)
    if len(xi_players) == engine.STARTING_XI_SIZE:
        from collections import Counter
        pos = Counter(p.position for p in xi_players)
        lo_hi = engine.FORMATION_BOUNDS
        legal = all(lo_hi[k][0] <= pos.get(k, 0) <= lo_hi[k][1] for k in lo_hi)
        if legal:
            scored = engine.expected_points(xi_players, formula)
            xi_cards = [engine._player_line(r) for _, r in scored.iterrows()]
            xi_names = {p.name for p in xi_players}
            bench_players = [p for p in squad.players if p.name not in xi_names]
            bench_scored = engine.expected_points(bench_players, formula) if bench_players else None
            bench_cards = ([engine._player_line(r) for _, r in bench_scored.iterrows()]
                           if bench_scored is not None else [])
            out["best_xi"] = xi_cards
            out["bench"] = bench_cards
            out["formation"] = f"{pos.get('DEF',0)}-{pos.get('MID',0)}-{pos.get('FWD',0)}"
    return out


# ---------------------------------------------------------------------------
# initial squad
# ---------------------------------------------------------------------------

# How many top players per position to hand the LLM. Big enough to cover every
# realistic FPL pick (the rest of the ~800-player pool is never-plays fodder),
# small enough to keep the prompt affordable. Dossiers are compact (news only
# attached when present), so this stays well within the model's context.
SHORTLIST_SIZES = {"GK": 15, "DEF": 65, "MID": 70, "FWD": 35}


def _shortlist(pool: list[PlayerFeatures], formula: str) -> list[PlayerFeatures]:
    scored = engine.expected_points(pool, formula)
    scored["_obj"] = pool
    out = []
    for pos, n in SHORTLIST_SIZES.items():
        sub = scored[scored["position"] == pos].sort_values("expected_score", ascending=False)
        out.extend(sub.head(n)["_obj"].tolist())
    return out


def recommend_squad(
    pool: list[PlayerFeatures],
    budget: float = 100.0,
    mode: str = "stats",
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    llm_config: Optional[LLMConfig] = None,
    news_provider: Optional[NewsProvider] = None,
) -> dict:
    stats = engine.stats_squad(pool, budget=budget, formula=formula)
    if mode == "stats" or "error" in stats:
        return {"mode": "stats", **stats}

    shortlist = _shortlist(pool, formula)
    attach_news(shortlist, news_provider or NullNewsProvider())
    user = prompts.squad_prompt(shortlist, budget)
    parsed, model, err = _run_llm(prompts.SYSTEM, user, llm_config)

    if parsed is None:
        return {"mode": "llm", "llm_failed": True, "llm_error": err,
                "fallback": "stats", **stats}
    return finalize_squad(pool, parsed, model, budget=budget, formula=formula,
                          llm_config=llm_config, stats=stats)


def finalize_squad(
    pool: list[PlayerFeatures],
    parsed: dict,
    model: Optional[str],
    budget: float = 100.0,
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    llm_config: Optional[LLMConfig] = None,
    stats: Optional[dict] = None,
) -> dict:
    """Turn an LLM's parsed squad JSON into a final, always-legal squad
    (BYOK-callable). llm_config may be None - reasoning regeneration then uses
    the deterministic fallback."""
    stats = stats or engine.stats_squad(pool, budget=budget, formula=formula)

    names = parsed.get("squad", [])
    picked = _resolve_names(names, pool)
    prefer = {p.name for p in picked}

    # Always render a LEGAL squad that includes the AI's picks as far as the
    # rules allow. If the AI's 15 were already legal, this reproduces them
    # exactly; if not, it keeps the AI's players it can and fills the rest with
    # the best legal options - so the pitch always matches the AI's intent as
    # closely as possible and is never illegal or blank.
    guided = engine.stats_squad(pool, budget=budget, formula=formula, prefer_names=prefer)
    guided_names = {p["name"] for p in guided.get("squad", [])}
    kept = sorted(n for n in prefer if n in guided_names)
    dropped = sorted(n for n in prefer if n not in guided_names)

    xi_names = {p["name"] for p in guided.get("best_xi", [])}
    llm_cap, llm_vice = parsed.get("captain"), parsed.get("vice_captain")
    # resolve AI captain/vice to a real name in the XI if possible
    cap = _match_in(llm_cap, xi_names) or guided.get("captain")
    vice = _match_in(llm_vice, xi_names) or guided.get("vice_captain")

    # If we had to change the AI's squad (dropped picks / different players),
    # the AI's original prose no longer matches the pitch. Regenerate the
    # reasoning FROM THE FINAL squad so text and picture always agree.
    reasoning = parsed.get("reasoning")
    if dropped:
        reasoning = _squad_reasoning_for(guided, cap, dropped, llm_config)

    return {
        "mode": "llm",
        "model": model,
        "squad": [p["name"] for p in guided.get("squad", [])],
        "best_xi": guided.get("best_xi", []),
        "bench": guided.get("bench", []),
        "formation": guided.get("formation"),
        "captain": cap,
        "vice_captain": vice,
        "total_cost": guided.get("total_cost"),
        "expected_xi_points": guided.get("expected_xi_points"),
        "overrode_stats": parsed.get("overrode_stats"),
        "reasoning": reasoning,
        "ai_players_kept": kept,
        "ai_players_dropped": dropped,
        "stats_anchor": stats,
    }


def _squad_reasoning_for(guided: dict, captain: str, dropped: list,
                         llm_config: Optional[LLMConfig]) -> str:
    """Write reasoning that describes the FINAL (already-legal) squad, so the
    text matches the pitch. Tries the LLM for a natural explanation; falls back
    to a deterministic summary if the call fails."""
    xi = guided.get("best_xi", [])
    bench = guided.get("bench", [])

    def line(p):
        return f"{p['name']} ({p['position']}, £{p['price']}m)"
    squad_txt = "Starting XI: " + ", ".join(line(p) for p in xi) + \
                ". Bench: " + ", ".join(line(p) for p in bench) + "."

    try:
        client = LLMClient(llm_config)
        system = ("You are an FPL analyst. In 2-3 sentences, explain the logic of "
                  "the given squad and captain. Describe ONLY the players listed - "
                  "do not mention any other player. Plain text, no JSON.")
        user = (f"Final squad (already legal, £{guided.get('total_cost')}m). "
                f"Captain: {captain}.\n{squad_txt}")
        text = client.complete(system, user).strip()
        if text:
            return text
    except Exception:
        pass

    # deterministic fallback - describes ONLY the final squad
    core = sorted(xi, key=lambda p: p.get("expected_score", 0), reverse=True)[:3]
    return (f"Legal £{guided.get('total_cost')}m squad captained by {captain}, built "
            f"around {', '.join(p['name'] for p in core)}, with cheap bench enablers "
            f"({', '.join(p['name'] for p in bench)}).")


def _match_in(name: str | None, xi_names: set) -> str | None:
    """Return the XI name matching `name` (exact or by surname), else None."""
    if not name:
        return None
    if name in xi_names:
        return name
    sn = name.strip().lower().split()[-1]
    for x in xi_names:
        if x.lower().split()[-1] == sn:
            return x
    return None


def _resolve_names(names: list[str], pool: list[PlayerFeatures]) -> list[PlayerFeatures]:
    """Map LLM-provided player names (often short, e.g. 'Saka', 'Calvert-Lewin')
    to actual pool players. Tries exact full name, then surname, then a
    substring match, never reusing a player twice."""
    full = {p.name.strip().lower(): p for p in pool}
    surname: dict[str, list[PlayerFeatures]] = {}
    for p in pool:
        surname.setdefault(p.name.split()[-1].lower(), []).append(p)

    resolved, used = [], set()
    for raw in names or []:
        key = str(raw).strip().lower()
        if not key:
            continue
        p = full.get(key)
        if p is None:                                   # surname match
            cands = [c for c in surname.get(key.split()[-1], []) if c.name not in used]
            p = cands[0] if cands else None
        if p is None:                                   # substring either way
            for c in pool:
                cl = c.name.lower()
                if c.name not in used and (key in cl or cl.split()[-1] in key):
                    p = c
                    break
        if p is not None and p.name not in used:
            resolved.append(p)
            used.add(p.name)
    return resolved


def _validate_llm_squad(players: list[PlayerFeatures], budget: float) -> list[str]:
    """Return a list of rule violations for an LLM-proposed squad (empty = ok)."""
    problems = []
    if len(players) != engine.SQUAD_SIZE:
        problems.append(f"has {len(players)} players, need {engine.SQUAD_SIZE}")
    by_pos, by_club, cost = {}, {}, 0.0
    for p in players:
        by_pos[p.position] = by_pos.get(p.position, 0) + 1
        by_club[p.team] = by_club.get(p.team, 0) + 1
        cost += p.price
    for pos, need in engine.SQUAD_QUOTA.items():
        if by_pos.get(pos, 0) != need:
            problems.append(f"{pos}: {by_pos.get(pos, 0)}/{need}")
    over = [c for c, n in by_club.items() if n > engine.MAX_PER_CLUB]
    if over:
        problems.append(f"too many from: {', '.join(over)}")
    if cost > budget + 1e-6:
        problems.append(f"over budget (£{cost:.1f}m > £{budget:.0f}m)")
    return problems


# ---------------------------------------------------------------------------
# transfers
# ---------------------------------------------------------------------------

def recommend_transfers(
    squad: Squad,
    candidate_pool: list[PlayerFeatures],
    mode: str = "stats",
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    llm_config: Optional[LLMConfig] = None,
    news_provider: Optional[NewsProvider] = None,
    max_suggestions: int = 5,
) -> dict:
    stats = engine.stats_transfers(squad, candidate_pool, formula=formula,
                                   max_suggestions=max_suggestions)
    if mode == "stats":
        return {"mode": "stats", **stats}

    provider = news_provider or NullNewsProvider()
    attach_news(squad.players, provider)
    attach_news(candidate_pool, provider)
    user = prompts.transfers_prompt(
        squad.players, candidate_pool, squad.bank, squad.free_transfers
    )
    parsed, model, err = _run_llm(prompts.SYSTEM, user, llm_config)

    if parsed is None:
        return {"mode": "llm", "llm_failed": True, "llm_error": err,
                "fallback": "stats", **stats}
    return finalize_transfers(squad, candidate_pool, parsed, model, formula,
                              max_suggestions=max_suggestions, stats=stats)


def finalize_transfers(
    squad: Squad,
    candidate_pool: list[PlayerFeatures],
    parsed: dict,
    model: Optional[str],
    formula: str = engine.DEFAULT_SCORING_FORMULA,
    max_suggestions: int = 5,
    stats: Optional[dict] = None,
) -> dict:
    """Turn an LLM's parsed transfers JSON into the final result (BYOK-callable)."""
    stats = stats or engine.stats_transfers(squad, candidate_pool, formula=formula,
                                            max_suggestions=max_suggestions)
    return {
        "mode": "llm",
        "model": model,
        "transfers": parsed.get("transfers", []),
        "take_hit": parsed.get("take_hit"),
        "overrode_stats": parsed.get("overrode_stats"),
        "reasoning": parsed.get("reasoning"),
        "stats_anchor": stats,
    }
