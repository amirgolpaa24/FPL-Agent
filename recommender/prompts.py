"""
Prompt construction for the LLM decision mode.

Design: give the model an UNBIASED picture. It is NOT shown the stats model's
pick, so it reasons independently from the raw data + news rather than
anchoring to a precomputed answer. Each player carries their key numbers
(including FPL's own official expected points), and any news is attached with
its source/reliability/corroboration/recency. High-impact news is also pulled
into a KEY NEWS block at the top so it can't be missed in a long list.
"""

from __future__ import annotations

from .features import PlayerFeatures, NewsItem

SYSTEM = (
    "You are an expert Fantasy Premier League analyst. You reason independently "
    "from the data and news given - there is no precomputed answer to anchor to. "
    "You combine hard statistics (expected points, season & recent form, minutes, "
    "fixtures, price/value) with real-world context (injuries, rotation, "
    "transfers, manager comments, form narratives) to make specific, decisive "
    "recommendations.\n\n"
    "Each news item is annotated with its SOURCE, a RELIABILITY score (0-100%), "
    "how many independent SOURCES corroborate it, and its RECENCY. Weigh news "
    "accordingly: trust official and high-reliability, multi-source, recent, "
    "clearly performance-relevant items (availability, minutes, suspensions, "
    "returns); heavily discount single-source low-reliability rumours and stale "
    "or vague mentions. A 100%-reliability official injury flag outweighs a lone "
    "tabloid rumour, even if the rumour is exciting.\n\n"
    "Obey all squad/formation rules exactly. You ALWAYS reply with a single "
    "valid JSON object and nothing else."
)


def _f(v) -> str:
    return "n/a" if v is None else f"{float(v):.2f}"


def _player_dossier(p: PlayerFeatures) -> str:
    ep = f"xP={_f(p.fpl_ep_next)}, " if p.fpl_ep_next is not None else ""
    avail = f", avail={_f(p.availability)}" if p.availability is not None else ""
    stat = (
        f"{p.name} ({p.position}, {p.team}, £{p.price:.1f}m) | "
        f"{ep}season_avg={_f(p.season_form_points)}, form={_f(p.form_points)}, "
        f"mins={_f(p.form_minutes)}, FDR={_f(p.fixture_difficulty)}/5, home={p.was_home}{avail}"
    )
    # Only attach a news block when the player actually has news - keeps the
    # prompt compact so we can afford to include many more players.
    if p.news:
        return stat + "\n    news: " + " | ".join(n.as_line() for n in p.news)
    return stat


def _dossiers(players: list[PlayerFeatures]) -> str:
    return "\n".join(f"- {_player_dossier(p)}" for p in players)


def key_news_block(players: list[PlayerFeatures], top_n: int = 25) -> str:
    """Pull the highest-impact news items across the players into a prominent
    block, so important context isn't buried in a long candidate list."""
    items: list[tuple[float, str, NewsItem]] = []
    for p in players:
        for n in p.news or []:
            score = (n.reliability or 0) * (1 + 0.3 * (n.corroboration or 1))
            items.append((score, p.name, n))
    if not items:
        return "KEY NEWS: (none gathered)\n"
    items.sort(key=lambda x: x[0], reverse=True)
    lines = [f"  - {name}: {n.as_line()}" for _, name, n in items[:top_n]]
    return "KEY NEWS (weigh by reliability / corroboration / recency):\n" + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# captain
# ---------------------------------------------------------------------------

def captain_prompt(candidates: list[PlayerFeatures]) -> str:
    return f"""Choose the single best CAPTAIN (and vice-captain) for the upcoming
gameweek. The captain's points are doubled, so prioritise a high, reliable
ceiling this week. Reason independently from the data and news below.

{key_news_block(candidates)}
Candidates:
{_dossiers(candidates)}

Return JSON exactly like:
{{
  "captain": "<player name>",
  "vice_captain": "<player name>",
  "confidence": <0-1>,
  "key_factors": ["<short factor>", "..."],
  "reasoning": "<2-3 sentences: why this captain, citing any decisive news>"
}}"""


# ---------------------------------------------------------------------------
# lineup
# ---------------------------------------------------------------------------

def lineup_prompt(squad_players: list[PlayerFeatures]) -> str:
    return f"""Choose the STARTING XI from this 15-man squad for the upcoming gameweek.
Rules: exactly 11 players; exactly 1 goalkeeper; 3-5 defenders; 2-5 midfielders;
1-3 forwards. Bench the other 4. Also name a captain and vice-captain from the XI.
Reason independently from the data and news below.

{key_news_block(squad_players)}
Squad:
{_dossiers(squad_players)}

Return JSON exactly like:
{{
  "formation": "<e.g. 3-4-3>",
  "starting_xi": ["<name>", ... 11 names],
  "bench_order": ["<name>", "<name>", "<name>", "<name>"],
  "captain": "<name>",
  "vice_captain": "<name>",
  "reasoning": "<2-3 sentences, citing any news that shaped the XI or bench order>"
}}"""


# ---------------------------------------------------------------------------
# initial squad
# ---------------------------------------------------------------------------

def squad_prompt(shortlist: list[PlayerFeatures], budget: float) -> str:
    return f"""Build a complete 15-man Fantasy Premier League squad for the START of the
season under a £{budget:.0f}m budget. Reason independently from the data and news
below - there is no preset answer.

Hard rules (a squad breaking ANY of these is invalid - obey exactly):
- Exactly 15 players: EXACTLY 2 goalkeepers, 5 defenders, 5 midfielders, 3 forwards.
- Total cost must not exceed £{budget:.0f}m.
- At most 3 players from any single Premier League club.

Strategy: spend big on a strong starting XI, use cheap enablers for the bench
and backup keeper. Weigh season-long scoring, expected points (xP), price/value,
opening fixtures, and the news (a big-name signing who's injured, or a nailed-on
starter at a promoted club, can beat a numbers-only pick).

{key_news_block(shortlist)}
Available players (a broad candidate list across every position):
{_dossiers(shortlist)}

Double-check your squad has exactly 2 GK, 5 DEF, 5 MID, 3 FWD, <=3 per club, and
fits the budget BEFORE replying. Return JSON exactly like:
{{
  "squad": ["<15 player names>"],
  "starting_xi": ["<11 names from the squad>"],
  "bench_order": ["<4 names>"],
  "captain": "<name>",
  "vice_captain": "<name>",
  "formation": "<e.g. 3-4-3>",
  "estimated_cost": <number>,
  "reasoning": "<3-4 sentences; cite the news that shaped any non-obvious picks>"
}}"""


# ---------------------------------------------------------------------------
# transfers
# ---------------------------------------------------------------------------

def transfers_prompt(
    squad_players: list[PlayerFeatures],
    pool: list[PlayerFeatures],
    bank: float,
    free_transfers: int,
) -> str:
    return f"""Recommend TRANSFERS for the upcoming gameweek. The manager has
£{bank:.1f}m in the bank and {free_transfers} free transfer(s); each extra transfer
costs -4 points. Suggest 0 to 3 swaps - recommending none ("hold") is valid and
often correct. A transfer must be same-position, affordable, and keep to max 3
players per club. Reason independently from the data and news below.

{key_news_block(squad_players + pool)}
Current squad:
{_dossiers(squad_players)}

Transfer targets available:
{_dossiers(pool)}

Return JSON exactly like:
{{
  "transfers": [
    {{"out": "<name>", "in": "<name>", "reason": "<short>"}}
  ],
  "take_hit": <true|false>,
  "reasoning": "<2-3 sentences citing the news/stats that justify each move>"
}}"""
