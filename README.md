# ⚽ FPL Agent

An AI-assisted Fantasy Premier League companion: builds your initial 15-man
squad, picks your weekly captain, sets your starting XI, and suggests
transfers — combining hard stats with automatically gathered,
reliability-weighted football news.

Installable on iPhone as a PWA, runs on a **100% free stack**, and uses a
**bring-your-own-key** AI model where your API key never leaves your device.

## 🌐 Live app

**[https://fpl-agent-myin.onrender.com](https://fpl-agent-myin.onrender.com)**

📱 On iPhone: open the link in Safari → Share → **Add to Home Screen** to
install it as an app.

> ⏳ First load can take ~30–60 seconds if the app is waking up (free hosting
> sleeps when idle) — after that it's fast.

## What it does

| Decision | What you get |
|---|---|
| 🧢 **Initial Squad** | A legal £100m 15 (2 GK / 5 DEF / 5 MID / 3 FWD, ≤3 per club), rendered on an FPL-style pitch with official kits |
| 🧠 **Captain** | Captain + vice for the gameweek, ranked candidates with expected points |
| 📋 **Starting XI** | Best legal formation + bench order from your actual squad |
| 🔁 **Transfers** | Swaps that raise expected points, net of the −4 hit; "hold" when that's optimal |

Every decision has two modes:

- **📊 Stats** — deterministic engine over FPL's official expected points
  (`ep_next`), season/recent form, ICT, fixture difficulty, availability and
  price. Backtested against 6 historical seasons. Free, no key needed.
- **🤖 AI** — the same numbers **plus auto-gathered news** (official FPL
  injury flags, BBC / Guardian / Sky feeds), each item weighted by source
  reliability, cross-source corroboration and recency, reasoned over by an
  LLM. Results are validated server-side so the AI can never produce an
  illegal squad.

## The BYOK trust model

AI mode uses **your own** OpenAI or Anthropic key (⚙️ in the app):

- stored **only in your browser/device storage**,
- sent **directly from your device to the provider**,
- **never** transmitted to or stored on this project's servers — verify it
  yourself in the network tab.

## Architecture

```
iPhone / browser (PWA) ──► FastAPI (Render, free) ──► SQLite / Supabase Postgres
        │                        ▲
        │ AI calls direct        │ fresh data + scored news
        ▼                        │
  OpenAI / Anthropic     GitHub Actions crawler (free cron,
  (user's own key)       ramps up before GW deadlines)
```

- `recommender/` — the decision engine: multi-factor scoring, squad/lineup
  optimizers, news aggregation, LLM prompts + validation, BYOK prepare/finalize
- `api.py` — FastAPI app: web UI, `/api/*` endpoints, PWA assets
- `web/` — single-page app (no framework), pitch visuals, settings, auth
- `scripts/` — data fetchers, dataset builder, backtests, news crawler
- `supabase/schema.sql` — Postgres schema with row-level security
- `dashboard.py` + `pages/` — Streamlit analytics workbench (backtesting,
  formula comparison, transfer signals) for development/analysis

## Quick start (local)

```bash
git clone https://github.com/<you>/FPL-Agent && cd FPL-Agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
# open http://localhost:8000  (live FPL data auto-fetches on startup)
```

Optional: link your FPL team (`python3 scripts/fetch_live.py --entry <team-id>`),
build the local DB (`python3 scripts/init_db.py`), or explore the analytics
dashboard (`streamlit run dashboard.py`).

## Deploying (free)

See **[DEPLOYMENT.md](DEPLOYMENT.md)** — ~30 minutes: Render (hosting),
Supabase (Postgres + magic-link accounts), GitHub Actions (scheduled news
crawler), then *Add to Home Screen* on iPhone.

## Data sources & credits

- [Official FPL API](https://fantasy.premierleague.com/api/bootstrap-static/) —
  live players, prices, fixtures, official expected points & availability
- [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) —
  historical season archives used for backtesting
- BBC Sport / The Guardian / Sky Sports RSS — news signals

Not affiliated with the Premier League. FPL data belongs to its owners; this
is a fan-made tool.
