"""
FPL Agent recommender: decision engines for the three core FPL choices -
picking the starting XI, choosing a captain, and suggesting transfers.

Two modes for every decision:
  - "stats": deterministic, based on the best-performing formulas found via
    backtesting (see scripts/compare_formulas.py). Fast, explainable, offline.
  - "llm": feeds all the same stats PLUS free-text news/injury/rumour context
    to a language model, which returns a judgement the pure-stats path can't -
    because it can weigh external factors the historical data never contains.

The core engine (features, engine, llm, news, prompts, recommenders) depends
only on the stdlib + pandas, so it's fully unit-testable without a web server.
FastAPI/pydantic live only in the api.py layer on top of this package.
"""

from .features import PlayerFeatures, NewsItem, Squad  # noqa: F401
from .recommenders import (  # noqa: F401
    recommend_captain,
    recommend_lineup,
    recommend_transfers,
    recommend_squad,
)

__all__ = [
    "PlayerFeatures",
    "NewsItem",
    "Squad",
    "recommend_captain",
    "recommend_lineup",
    "recommend_transfers",
    "recommend_squad",
]
