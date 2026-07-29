"""
Plain-dataclass data model shared by the stats and LLM engines.

These are intentionally NOT pydantic models - keeping the core engine free of
web-framework dependencies means it can be imported and unit-tested anywhere.
The FastAPI layer (api.py) defines matching pydantic schemas and converts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional

VALID_POSITIONS = ("GK", "DEF", "MID", "FWD")


@dataclass
class NewsItem:
    """A single piece of external context about a player or their team -
    the kind of thing the pure-stats path structurally cannot see."""
    headline: str
    detail: str = ""
    source: str = ""
    # Free-form category, e.g. "injury", "transfer", "rotation", "form",
    # "suspension", "rumour". Not validated - the LLM reads it as a hint.
    category: str = "general"
    # Optional caller-supplied sentiment in [-1, 1]: negative = bad for the
    # player's expected points, positive = good. Purely advisory.
    sentiment: Optional[float] = None
    # Source trustworthiness in [0, 1] (official > major outlet > tabloid/rumour)
    # and how many independent sources corroborate this item. The LLM is told
    # to weigh news by these.
    reliability: Optional[float] = None
    corroboration: int = 1
    recency: str = ""          # e.g. "2h ago", "today", "" if unknown

    def as_line(self) -> str:
        bits = [f"[{self.category}]", self.headline.strip()]
        if self.detail:
            bits.append(f"- {self.detail.strip()}")
        meta = []
        if self.source:
            meta.append(self.source)
        if self.reliability is not None:
            meta.append(f"reliability {self.reliability:.0%}")
        if self.corroboration and self.corroboration > 1:
            meta.append(f"{self.corroboration} sources")
        if self.recency:
            meta.append(self.recency)
        if meta:
            bits.append("(" + ", ".join(meta) + ")")
        return " ".join(bits)


@dataclass
class PlayerFeatures:
    """Everything known about a player going INTO a gameweek. Only
    pre-kickoff information belongs here - never that gameweek's outcome."""
    name: str
    team: str
    position: str            # one of VALID_POSITIONS
    price: float             # in £m, e.g. 12.5

    # Pre-gameweek predictive features (all optional; missing -> treated as 0
    # or neutral by the engine). Names mirror the processed feature table.
    form_points: Optional[float] = None            # recent points, rolling avg
    season_form_points: Optional[float] = None     # season-long avg (strongest signal)
    last_gw_points: Optional[float] = None
    form_ict: Optional[float] = None
    form_minutes: Optional[float] = None           # recent minutes, rolling avg
    fixture_difficulty: Optional[float] = None      # official FDR, 1 (easy)..5 (hard)
    opp_goals_conceded_form: Optional[float] = None
    was_home: Optional[bool] = None
    availability: Optional[float] = None            # P(features), 0-1 (from live chance-of-playing)
    fpl_ep_next: Optional[float] = None             # FPL's OFFICIAL expected points, next GW

    # Identity / bookkeeping
    player_id: Optional[int] = None
    team_code: Optional[int] = None   # FPL team code, for jersey/badge images
    gameweek: Optional[int] = None

    # Attached external context (used by the LLM path; ignored by stats path)
    news: list[NewsItem] = field(default_factory=list)

    def __post_init__(self):
        if self.position not in VALID_POSITIONS:
            raise ValueError(
                f"{self.name}: position must be one of {VALID_POSITIONS}, got {self.position!r}"
            )
        if self.price is None or self.price < 0:
            raise ValueError(f"{self.name}: price must be a non-negative number")

    def to_row(self) -> dict:
        """Flat dict for building a pandas DataFrame the formulas can score."""
        d = asdict(self)
        d.pop("news", None)
        return d

    def news_block(self) -> str:
        if not self.news:
            return "(no external news provided)"
        return "\n".join(f"  - {n.as_line()}" for n in self.news)


@dataclass
class Squad:
    """A 15-man FPL squad plus available bank, used for lineup & transfers."""
    players: list[PlayerFeatures]
    bank: float = 0.0          # money in the bank, £m
    free_transfers: int = 1

    # Standard FPL squad composition
    QUOTA = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    MAX_PER_CLUB = 3
    SIZE = 15

    def validate(self) -> list[str]:
        """Return a list of rule violations (empty = a legal squad). We warn
        rather than hard-raise, so callers can still get recommendations for a
        slightly-off squad during testing."""
        problems = []
        if len(self.players) != self.SIZE:
            problems.append(f"squad has {len(self.players)} players, expected {self.SIZE}")
        by_pos = {}
        by_club = {}
        for p in self.players:
            by_pos[p.position] = by_pos.get(p.position, 0) + 1
            by_club[p.team] = by_club.get(p.team, 0) + 1
        for pos, need in self.QUOTA.items():
            if by_pos.get(pos, 0) != need:
                problems.append(f"{pos}: have {by_pos.get(pos, 0)}, need {need}")
        for club, n in by_club.items():
            if n > self.MAX_PER_CLUB:
                problems.append(f"{club}: {n} players exceeds max {self.MAX_PER_CLUB}")
        return problems
