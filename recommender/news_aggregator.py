"""
Automatic news aggregation. Instead of the user typing news by hand, the app
gathers recent, performance-relevant news for the players in the pool from
multiple sources, scores each item, and hands the weighted set to the AI.

Sources (each with a reliability weight):
  1. FPL Official  - the `news`/`status`/chance-of-playing fields in the FPL
     API. Authoritative for availability; needs no crawling. Always on.
  2. Football RSS  - reputable outlets (BBC Sport, Guardian, Sky Sports).
     Fetched live; matched to players by surname. Graceful on failure.
  3. NewsAPI        - optional, only if NEWS_API_KEY is set (broader coverage).

Scoring the AI is told to use:
  - source reliability (assigned per source below),
  - corroboration: how many independent sources carry the same player story,
  - recency: newer items matter more,
  - performance relevance: items about injuries, minutes, suspensions, returns,
    transfers weigh more than generic mentions.

Network calls run on the user's machine (which has internet). Every fetch is
wrapped so a dead feed or timeout never breaks a recommendation - at worst you
fall back to FPL-official news only.
"""

from __future__ import annotations

import os
import re
from typing import Optional
from xml.etree import ElementTree as ET

import requests

from .features import NewsItem

# per-source trustworthiness in [0,1]
SOURCE_RELIABILITY = {
    "FPL Official": 1.00,
    "FPL API": 1.00,       # official availability flags = authoritative
    "BBC Sport": 0.90,
    "The Guardian": 0.85,
    "Sky Sports": 0.80,
    "NewsAPI": 0.60,
}

RSS_FEEDS = {
    "BBC Sport": "https://feeds.bbci.co.uk/sport/football/rss.xml",
    "The Guardian": "https://www.theguardian.com/football/rss",
    "Sky Sports": "https://www.skysports.com/rss/12040",
}

# words that make a headline performance-relevant (and a rough category)
RELEVANCE_KEYWORDS = {
    "injury": ["injury", "injured", "knock", "strain", "fitness", "doubt", "sidelined", "ruled out", "hamstring", "ankle", "back"],
    "suspension": ["suspended", "suspension", "banned", "red card", "sent off"],
    "rotation": ["rested", "benched", "rotation", "dropped", "left out", "rotated"],
    "return": ["returns", "return", "back in", "fit again", "available", "recovered"],
    "transfer": ["signs", "signed", "transfer", "joins", "move", "deal", "loan"],
    "form": ["hat-trick", "brace", "starring", "star", "in form", "goal", "assist"],
}


def _classify(text: str) -> Optional[str]:
    t = text.lower()
    for cat, words in RELEVANCE_KEYWORDS.items():
        if any(w in t for w in words):
            return cat
    return None


def _surname(full_name: str) -> str:
    parts = full_name.split()
    return parts[-1] if parts else full_name


def fetch_rss(url: str, timeout: int = 8) -> list[dict]:
    """Return [{title, summary, published}] for a feed. Never raises."""
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "fpl-agent/1.0"})
        if resp.status_code >= 400:
            return []
        root = ET.fromstring(resp.content)
    except Exception:
        return []
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        summary = (it.findtext("description") or "").strip()
        pub = (it.findtext("pubDate") or "").strip()
        if title:
            items.append({"title": title, "summary": summary, "published": pub})
    return items


def _match_players(entry_text: str, surnames: dict[str, list[str]]) -> list[str]:
    """Return player names whose surname appears as a whole word in the text."""
    text = entry_text.lower()
    hits = []
    for sur, names in surnames.items():
        if len(sur) < 4:
            continue
        if re.search(r"\b" + re.escape(sur) + r"\b", text):
            hits.extend(names)
    return hits


def gather(players, max_players: Optional[int] = None, use_rss: bool = True) -> dict[str, list[NewsItem]]:
    """Collect news per player name. `players` is a list of PlayerFeatures.
    All players are matched against fetched news by default; pass max_players
    to cap the RSS-matched set. FPL-official news is attached for everyone."""
    by_name: dict[str, list[NewsItem]] = {}

    # ---- 1. FPL official (already on each player as auto-news) --------------
    for p in players:
        for n in getattr(p, "news", []) or []:
            n.source = n.source or "FPL Official"
            n.reliability = SOURCE_RELIABILITY.get(n.source, n.reliability or 0.9)
            by_name.setdefault(p.name, []).append(n)

    if not use_rss:
        return by_name

    # ---- 2. RSS outlets -----------------------------------------------------
    subset = players if max_players is None else players[:max_players]
    surnames: dict[str, list[str]] = {}
    for p in subset:
        surnames.setdefault(_surname(p.name).lower(), []).append(p.name)

    # count how many distinct sources mention each player (corroboration)
    mentions: dict[str, set] = {}
    staged: dict[str, list[NewsItem]] = {}

    for source, url in RSS_FEEDS.items():
        rel = SOURCE_RELIABILITY.get(source, 0.6)
        for entry in fetch_rss(url):
            blob = f"{entry['title']} {entry['summary']}"
            cat = _classify(blob)
            if cat is None:
                continue  # skip non-performance-relevant football news
            for name in set(_match_players(blob, surnames)):
                mentions.setdefault(name, set()).add(source)
                staged.setdefault(name, []).append(NewsItem(
                    headline=entry["title"], detail=entry["summary"][:200],
                    source=source, category=cat, reliability=rel,
                    recency=_relative_time(entry.get("published", "")),
                ))

    # ---- 3. optional NewsAPI ------------------------------------------------
    api_key = os.getenv("NEWS_API_KEY")
    if api_key:
        _newsapi(subset, surnames, staged, mentions, api_key)

    # attach corroboration counts and merge
    for name, items in staged.items():
        n_sources = len(mentions.get(name, set()))
        for it in items:
            it.corroboration = max(1, n_sources)
        by_name.setdefault(name, []).extend(items)

    return by_name


def _newsapi(players, surnames, staged, mentions, api_key: str) -> None:
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params={"q": "Premier League injury OR lineup OR transfer",
                    "language": "en", "sortBy": "publishedAt", "pageSize": 50,
                    "apiKey": api_key},
            timeout=8,
        )
        if resp.status_code >= 400:
            return
        for a in resp.json().get("articles", []):
            blob = f"{a.get('title','')} {a.get('description','')}"
            cat = _classify(blob)
            if cat is None:
                continue
            for name in set(_match_players(blob, surnames)):
                mentions.setdefault(name, set()).add("NewsAPI")
                staged.setdefault(name, []).append(NewsItem(
                    headline=a.get("title", ""), detail=(a.get("description") or "")[:200],
                    source="NewsAPI", category=cat,
                    reliability=SOURCE_RELIABILITY["NewsAPI"],
                    recency=_relative_time(a.get("publishedAt", "")),
                ))
    except Exception:
        return


def _relative_time(pubdate: str) -> str:
    """Best-effort 'today'/'recent' label; robust to varied date formats."""
    if not pubdate:
        return ""
    import datetime as dt
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z",
                "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            when = dt.datetime.strptime(pubdate, fmt)
            if when.tzinfo:
                when = when.astimezone(dt.timezone.utc).replace(tzinfo=None)
            hrs = (dt.datetime.utcnow() - when).total_seconds() / 3600
            if hrs < 1:
                return "just now"
            if hrs < 24:
                return f"{int(hrs)}h ago"
            return f"{int(hrs/24)}d ago"
        except (ValueError, OverflowError):
            continue
    return ""


def flatten(by_name: dict[str, list[NewsItem]]) -> list[dict]:
    """Flat list for display in the UI."""
    out = []
    for name, items in by_name.items():
        for it in items:
            out.append({
                "player": name, "headline": it.headline, "source": it.source,
                "category": it.category, "reliability": it.reliability,
                "corroboration": it.corroboration, "recency": it.recency,
            })
    # most reliable + most corroborated first
    out.sort(key=lambda x: ((x["reliability"] or 0), x["corroboration"]), reverse=True)
    return out
