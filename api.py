"""
FastAPI service exposing the three FPL decision engines as REST endpoints.

Run locally:
    pip install fastapi uvicorn pydantic requests pandas
    uvicorn api:app --reload --port 8000
    # interactive docs at http://localhost:8000/docs

Every endpoint accepts mode = "stats" (deterministic, offline) or "llm"
(stats + caller-supplied news, judged by a language model). LLM provider/model
are configured per-request (falling back to server env vars), so you can pick
a model later without code changes.

This file is a thin translation layer: pydantic request models -> plain
dataclasses -> recommender package -> JSON. All real logic lives in the
recommender/ package, which has no web dependencies and is unit-tested
separately.
"""

from __future__ import annotations

from typing import Literal, Optional

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from recommender import engine, webservice
from recommender.features import PlayerFeatures, NewsItem, Squad
from recommender.llm import LLMConfig
from recommender.news import DictNewsProvider
from recommender.recommenders import (
    recommend_captain, recommend_lineup, recommend_transfers, recommend_squad,
)

app = FastAPI(
    title="FPL Agent Recommender API",
    version="1.0",
    description="Starting-XI, captaincy, and transfer recommendations - "
                "deterministic stats mode or news-aware LLM mode.",
)

_WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
_LIVE_BOOTSTRAP = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "live", "bootstrap.json")
_LIVE_MAX_AGE_HOURS = 6


@app.on_event("startup")
def _refresh_live_data_on_boot():
    """On a fresh deploy (e.g. Render) data/live/ doesn't exist, and on a
    long-running server it goes stale. If live data is missing or older than
    _LIVE_MAX_AGE_HOURS, fetch it in a background thread so boot isn't blocked
    and a failed fetch never breaks the app (it just runs historical-only)."""
    import threading
    import time

    def _refresh():
        try:
            age_ok = (os.path.exists(_LIVE_BOOTSTRAP)
                      and (time.time() - os.path.getmtime(_LIVE_BOOTSTRAP)) < _LIVE_MAX_AGE_HOURS * 3600)
            if age_ok:
                return
            import sys as _sys
            _scripts = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts")
            if _scripts not in _sys.path:
                _sys.path.insert(0, _scripts)
            from fetch_live import fetch_live, DEFAULT_OUT
            fetch_live(os.path.abspath(DEFAULT_OUT))
        except Exception:
            pass  # historical mode still works

    threading.Thread(target=_refresh, daemon=True).start()


# ---------------------------------------------------------------------------
# web app (single-page frontend) + its high-level endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    """Serve the single-page web app."""
    index = os.path.join(_WEB_DIR, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    raise HTTPException(404, "web/index.html not found")


def _static(name: str, media_type: str | None = None):
    path = os.path.join(_WEB_DIR, name)
    if os.path.exists(path):
        return FileResponse(path, media_type=media_type)
    raise HTTPException(404, f"{name} not found")


@app.get("/manifest.json")
def pwa_manifest():
    return _static("manifest.json", "application/manifest+json")


@app.get("/sw.js")
def pwa_sw():
    return _static("sw.js", "application/javascript")


@app.get("/icon-192.png")
def pwa_icon_192():
    return _static("icon-192.png", "image/png")


@app.get("/icon-512.png")
def pwa_icon_512():
    return _static("icon-512.png", "image/png")


@app.get("/config.js")
def frontend_config():
    return _static("config.js", "application/javascript")


@app.get("/api/status")
def api_status():
    """Initial state for the frontend: gameweek, data freshness, LLM config."""
    return webservice.status()


class RecommendRequest(BaseModel):
    decision: Literal["squad", "captain", "lineup", "transfers"]
    mode: Literal["stats", "llm"] = "stats"
    source: Optional[Literal["live", "historical"]] = None
    season: Optional[str] = None
    gameweek: Optional[int] = None
    budget: float = 100.0
    entry_id: Optional[str] = None
    bank: float = 2.0
    free_transfers: int = 1
    candidates: int = 15
    formula: Optional[str] = None
    news: list[dict] = []
    crawl_news: bool = True


@app.post("/api/recommend")
def api_recommend(req: RecommendRequest):
    """One endpoint the web app calls for every decision; builds data
    server-side and returns a browser-ready recommendation."""
    return webservice.recommend(req.model_dump())


class FinalizeRequest(BaseModel):
    payload: RecommendRequest
    llm_text: str
    model: Optional[str] = None


@app.post("/api/llm/prepare")
def api_llm_prepare(req: RecommendRequest):
    """BYOK step 1: build the prompt. The browser then calls the LLM provider
    DIRECTLY with the user's own key (which never touches this server)."""
    return webservice.prepare(req.model_dump())


@app.post("/api/llm/finalize")
def api_llm_finalize(req: FinalizeRequest):
    """BYOK step 2: validate the model's raw reply into a recommendation."""
    return webservice.finalize(req.payload.model_dump(), req.llm_text, req.model)


@app.get("/api/history")
def api_history(limit: int = 20):
    """Recently logged recommendations from the database."""
    try:
        import database as db
        return {"recommendations": db.recent_recommendations(limit).to_dict("records"),
                "database": db.stats()}
    except Exception as e:
        return {"recommendations": [], "error": str(e)}


# ---------------------------------------------------------------------------
# request / response schemas
# ---------------------------------------------------------------------------

class NewsItemIn(BaseModel):
    headline: str
    detail: str = ""
    source: str = ""
    category: str = "general"
    sentiment: Optional[float] = Field(None, ge=-1, le=1)

    def to_dc(self) -> NewsItem:
        return NewsItem(**self.model_dump())


class PlayerIn(BaseModel):
    name: str
    team: str
    position: Literal["GK", "DEF", "MID", "FWD"]
    price: float = Field(..., ge=0, description="Price in £m, e.g. 12.5")

    form_points: Optional[float] = None
    season_form_points: Optional[float] = None
    last_gw_points: Optional[float] = None
    form_ict: Optional[float] = None
    form_minutes: Optional[float] = None
    fixture_difficulty: Optional[float] = Field(None, ge=1, le=5)
    opp_goals_conceded_form: Optional[float] = None
    was_home: Optional[bool] = None
    player_id: Optional[int] = None
    gameweek: Optional[int] = None
    news: list[NewsItemIn] = []

    def to_dc(self) -> PlayerFeatures:
        data = self.model_dump()
        news = [NewsItemIn(**n).to_dc() for n in data.pop("news", [])]
        return PlayerFeatures(news=news, **data)


class LLMConfigIn(BaseModel):
    provider: Optional[Literal["openai", "anthropic", "ollama"]] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: Optional[float] = Field(None, ge=0, le=2)

    def to_cfg(self) -> LLMConfig:
        return LLMConfig.from_env(**{k: v for k, v in self.model_dump().items() if v is not None})


Mode = Literal["stats", "llm"]


class CaptainRequest(BaseModel):
    candidates: list[PlayerIn] = Field(..., min_length=1)
    mode: Mode = "stats"
    formula: str = engine.DEFAULT_SCORING_FORMULA
    llm_config: Optional[LLMConfigIn] = None


class LineupRequest(BaseModel):
    squad: list[PlayerIn] = Field(..., min_length=11)
    bank: float = 0.0
    free_transfers: int = 1
    mode: Mode = "stats"
    formula: str = engine.DEFAULT_SCORING_FORMULA
    llm_config: Optional[LLMConfigIn] = None


class SquadRequest(BaseModel):
    pool: list[PlayerIn] = Field(..., min_length=15,
                                 description="Candidate player pool to build the 15 from")
    budget: float = 100.0
    mode: Mode = "stats"
    formula: str = engine.DEFAULT_SCORING_FORMULA
    llm_config: Optional[LLMConfigIn] = None


class TransferRequest(BaseModel):
    squad: list[PlayerIn] = Field(..., min_length=11)
    candidate_pool: list[PlayerIn] = Field(..., min_length=1)
    bank: float = 0.0
    free_transfers: int = 1
    mode: Mode = "stats"
    formula: str = engine.DEFAULT_SCORING_FORMULA
    max_suggestions: int = 5
    llm_config: Optional[LLMConfigIn] = None


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _news_provider(players: list[PlayerFeatures]) -> DictNewsProvider:
    """Build a name-keyed provider from news embedded in the request players,
    so the LLM path sees exactly what the caller supplied."""
    return DictNewsProvider({p.name: p.news for p in players if p.news})


def _cfg(llm_config: Optional[LLMConfigIn]) -> Optional[LLMConfig]:
    return llm_config.to_cfg() if llm_config else None


def _validate_formula(formula: str):
    if formula not in engine.FORMULAS:
        raise HTTPException(422, f"Unknown formula {formula!r}. Options: {list(engine.FORMULAS)}")


# ---------------------------------------------------------------------------
# endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "formulas": list(engine.FORMULAS),
            "default_formula": engine.DEFAULT_SCORING_FORMULA}


@app.post("/recommend/captain")
def api_captain(req: CaptainRequest):
    _validate_formula(req.formula)
    candidates = [p.to_dc() for p in req.candidates]
    return recommend_captain(
        candidates, mode=req.mode, formula=req.formula,
        llm_config=_cfg(req.llm_config), news_provider=_news_provider(candidates),
    )


@app.post("/recommend/lineup")
def api_lineup(req: LineupRequest):
    _validate_formula(req.formula)
    players = [p.to_dc() for p in req.squad]
    squad = Squad(players=players, bank=req.bank, free_transfers=req.free_transfers)
    return recommend_lineup(
        squad, mode=req.mode, formula=req.formula,
        llm_config=_cfg(req.llm_config), news_provider=_news_provider(players),
    )


@app.post("/recommend/squad")
def api_squad(req: SquadRequest):
    _validate_formula(req.formula)
    pool = [p.to_dc() for p in req.pool]
    return recommend_squad(
        pool, budget=req.budget, mode=req.mode, formula=req.formula,
        llm_config=_cfg(req.llm_config), news_provider=_news_provider(pool),
    )


@app.post("/recommend/transfers")
def api_transfers(req: TransferRequest):
    _validate_formula(req.formula)
    squad_players = [p.to_dc() for p in req.squad]
    pool = [p.to_dc() for p in req.candidate_pool]
    squad = Squad(players=squad_players, bank=req.bank, free_transfers=req.free_transfers)
    return recommend_transfers(
        squad, pool, mode=req.mode, formula=req.formula,
        max_suggestions=req.max_suggestions, llm_config=_cfg(req.llm_config),
        news_provider=_news_provider(squad_players + pool),
    )
