# FPL Agent Recommender

Decision engines for the three core FPL choices — **starting XI**, **captain**,
and **transfers** — each available in two modes:

- **`stats`** — deterministic, offline, based on the formulas that won our
  backtests (`scripts/compare_formulas.py`). Default scorer:
  `expanding_season_form` (season-long form + official fixture difficulty).
- **`llm`** — feeds the *same* stats plus caller-supplied news/injury/rumour
  context to a language model, which returns a judgement the pure-stats path
  can't make. Always falls back to the stats result (flagged) if the LLM call
  or JSON parse fails.

## Layout

```
recommender/
  features.py      dataclasses: PlayerFeatures, NewsItem, Squad (no web deps)
  engine.py        stats logic: expected_points, stats_captain/lineup/transfers
  llm.py           provider-agnostic LLMClient (openai/anthropic/ollama adapters)
  news.py          pluggable NewsProvider interface + caller-provided providers
  prompts.py       LLM prompt templates (stats anchor + news + strict-JSON ask)
  recommenders.py  orchestration: recommend_captain/lineup/transfers (public API)
  data_adapter.py  build PlayerFeatures/Squad from processed data (tests + UI)
api.py             FastAPI service exposing the three endpoints
pages/8_🤖_Recommender.py   Streamlit UI over the same engine
```

The core package depends only on the stdlib + pandas, so it is unit-testable
without a web server or network. FastAPI/pydantic live only in `api.py`.

## Running the API

```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
# interactive docs: http://localhost:8000/docs
```

### Endpoints

| Endpoint               | Purpose                          |
|------------------------|----------------------------------|
| `GET  /health`         | liveness + available formulas    |
| `POST /recommend/squad`     | build the initial 15-man squad under budget |
| `POST /recommend/captain`   | best captain from candidates |
| `POST /recommend/lineup`    | best XI + bench + captain from a 15-man squad |
| `POST /recommend/transfers` | ranked transfer swaps        |

Every POST takes `mode: "stats" | "llm"`, an optional `formula`, an optional
`llm_config`, and news embedded per-player.

### Example: captain, stats mode

```bash
curl -s localhost:8000/recommend/captain -H 'content-type: application/json' -d '{
  "mode": "stats",
  "candidates": [
    {"name":"Mohamed Salah","team":"Liverpool","position":"MID","price":13.0,
     "season_form_points":8.9,"form_points":9.2,"form_minutes":88,"fixture_difficulty":2},
    {"name":"Erling Haaland","team":"Man City","position":"FWD","price":15.0,
     "season_form_points":7.6,"form_points":5.1,"form_minutes":90,"fixture_difficulty":3}
  ]
}'
```

### Example: captain, LLM mode with news

```bash
curl -s localhost:8000/recommend/captain -H 'content-type: application/json' -d '{
  "mode": "llm",
  "llm_config": {"provider":"openai","model":"gpt-4o"},
  "candidates": [
    {"name":"Mohamed Salah","team":"Liverpool","position":"MID","price":13.0,
     "season_form_points":8.9,"form_points":9.2,"form_minutes":88,"fixture_difficulty":2,
     "news":[{"headline":"Rested in midweek, fully fit","category":"rotation","sentiment":0.3}]},
    {"name":"Cole Palmer","team":"Chelsea","position":"MID","price":11.0,
     "season_form_points":8.1,"form_points":7.0,"form_minutes":90,"fixture_difficulty":2}
  ]
}'
```

## LLM configuration

Configured by env vars (overridable per request via `llm_config`):

| Var              | Meaning                                   | Default     |
|------------------|-------------------------------------------|-------------|
| `LLM_PROVIDER`   | `openai` \| `anthropic` \| `ollama`       | `openai`    |
| `LLM_MODEL`      | model id                                  | provider default |
| `LLM_API_KEY`    | api key (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`) | — |
| `LLM_BASE_URL`   | override endpoint (gateway / Ollama host) | provider default |
| `LLM_TEMPERATURE`| decoding temperature                      | `0.2`       |

No provider SDKs are required — adapters use `requests` directly.

## Using it live for the current season

The engine scores whatever features you give it; two adapters supply them:

- `data_adapter.py` — historical processed seasons (backtesting, demos, GW1
  cold-start testing).
- `live_fpl.py` — the **live current season** from the official FPL API.

### 1. Fetch live data (run locally, needs internet)

```bash
python3 scripts/fetch_live.py                # players, prices, fixtures, news
python3 scripts/fetch_live.py --entry 1234567  # also your own squad + bank
```

Saves to `data/live/`. Re-run before each gameweek for fresh prices/injuries.

### 2. Build the pool / your squad

```python
from recommender import live_fpl
pool  = live_fpl.build_pool(gameweek=1, prior_season="2025-26")
squad = live_fpl.build_squad_from_entry(1234567, prior_season="2025-26")
```

- **Cold-start:** at GW1 there's no within-season form, so it's seeded from the
  prior season's points-per-game (matched by name) and faded out by ~GW6. Pass
  the most recently completed season as `prior_season`.
- **Auto-news:** the API's own `status` / `news` / `chance_of_playing` fields
  become `NewsItem`s automatically, so LLM mode sees official injury/availability
  flags even before you add your own news.

### 3. Recommend

Same three functions as always (`recommend_captain/lineup/transfers`) — they
don't care whether the players came from history or the live API.

The Streamlit **🤖 Recommender** page exposes all of this with a live/historical
toggle and a gameweek selector (including GW1).

## News

News is **caller-provided**: attach `NewsItem`s per player in the request (or
via `DictNewsProvider`). To wire a live source later, implement the
`NewsProvider` protocol (`news_for(player) -> list[NewsItem]`) and pass it in —
no engine changes needed.
