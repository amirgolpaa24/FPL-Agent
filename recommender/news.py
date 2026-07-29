"""
Pluggable news interface. The caller supplies news (payload-driven, as chosen),
but everything is routed through a NewsProvider so a live fetcher can be dropped
in later without touching the engine.
"""

from __future__ import annotations

from typing import Protocol

from .features import NewsItem, PlayerFeatures


class NewsProvider(Protocol):
    """Anything that can return news items for a given player. Implement this
    to wire in a real source (an API, a scraper, an RSS reader, an LLM search
    tool) without changing the recommenders."""

    def news_for(self, player: PlayerFeatures) -> list[NewsItem]:
        ...


class NullNewsProvider:
    """Default: no external news. The LLM still runs, just on stats alone."""

    def news_for(self, player: PlayerFeatures) -> list[NewsItem]:
        return []


class DictNewsProvider:
    """Caller-provided news, keyed by player name (case-insensitive). This is
    the primary path: an app / notebook / API request hands us a mapping like
        {"Erling Haaland": [NewsItem(...)], "Mohamed Salah": [...]}
    and we attach it to the matching players."""

    def __init__(self, by_player: dict[str, list[NewsItem]]):
        self._by_player = {k.strip().lower(): v for k, v in (by_player or {}).items()}

    def news_for(self, player: PlayerFeatures) -> list[NewsItem]:
        return self._by_player.get(player.name.strip().lower(), [])


def attach_news(players: list[PlayerFeatures], provider: NewsProvider) -> list[PlayerFeatures]:
    """Populate each player's .news from a provider (in place, and returned)."""
    for p in players:
        fetched = provider.news_for(p)
        if fetched:
            # extend rather than replace, so request-embedded news is preserved
            existing = {n.as_line() for n in p.news}
            p.news.extend(n for n in fetched if n.as_line() not in existing)
    return players


def coerce_news(raw) -> list[NewsItem]:
    """Turn loosely-typed news (dicts from an API request) into NewsItem list."""
    items = []
    for entry in raw or []:
        if isinstance(entry, NewsItem):
            items.append(entry)
        elif isinstance(entry, dict):
            items.append(NewsItem(
                headline=entry.get("headline", entry.get("text", "")),
                detail=entry.get("detail", ""),
                source=entry.get("source", ""),
                category=entry.get("category", "general"),
                sentiment=entry.get("sentiment"),
            ))
        elif isinstance(entry, str):
            items.append(NewsItem(headline=entry))
    return items
