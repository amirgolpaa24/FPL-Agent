# Deploying FPL Agent — 100% free stack

Everything below uses free tiers only. Total cost: **$0** (users bring their
own AI key, which is the only paid piece anywhere).

```
┌─────────────┐   installs as app    ┌──────────────────────────────┐
│  iPhone /    │ ◄──────────────────  │  Render (free): FastAPI +    │
│  any browser │   PWA "Add to Home"  │  web app at your-app.onrender.com │
└──────┬───────┘                      └──────────────┬───────────────┘
       │  AI calls go DIRECTLY to                    │ reads/writes
       │  OpenAI/Anthropic with the                  ▼
       │  user's own key (BYOK) —      ┌──────────────────────────────┐
       │  never through the server     │  Supabase (free): Postgres + │
       ▼                               │  magic-link auth + RLS       │
┌──────────────┐                       └──────────────▲───────────────┘
│ LLM provider │                                      │ upserts fresh data
└──────────────┘                       ┌──────────────┴───────────────┐
                                       │  GitHub Actions (free cron): │
                                       │  news crawler before deadlines│
                                       └──────────────────────────────┘
```

## 1. Supabase (database + accounts) — ~10 minutes

1. Create a free account at [supabase.com](https://supabase.com) → **New project**.
2. In the dashboard: **SQL Editor → New query** → paste the whole of
   `supabase/schema.sql` → **Run**. This creates all tables, indexes, and
   Row-Level-Security policies (user data private; stats/news public-read).
3. **Settings → API**: copy the `Project URL` and the `anon` key into
   `web/config.js` (safe to ship — RLS protects everything). This turns on
   magic-link sign-in and cross-device FPL team ID memory.
4. Copy the `service_role` key too — that one is SECRET; it goes only into
   GitHub Actions secrets for the crawler (step 3).

## 2. Render (host the app) — ~10 minutes

1. Push this repo to GitHub.
2. Create a free account at [render.com](https://render.com) → **New → Blueprint**
   → connect the repo. `render.yaml` configures everything (free plan,
   `uvicorn api:app`).
3. Your app is live at `https://<name>.onrender.com`. Free services sleep when
   idle and wake in ~30s on the next visit.
4. Seed data once: from your machine run
   `python3 scripts/fetch_live.py && python3 scripts/init_db.py` and redeploy,
   or just let the GitHub Actions crawler populate Supabase.

## 3. GitHub Actions (news crawler) — ~5 minutes

1. In the GitHub repo: **Settings → Secrets and variables → Actions** and add:
   - `SUPABASE_URL` — from step 1
   - `SUPABASE_SERVICE_KEY` — from step 1 (secret!)
   - `NEWS_API_KEY` — optional, from newsapi.org (free tier)
2. The workflow `.github/workflows/news-crawler.yml` then runs automatically:
   every 6 hours, plus hourly on Friday/Saturday mornings UTC (deadline
   windows). It refreshes live FPL data and reliability-scored news into
   Supabase. You can also trigger it manually from the **Actions** tab.

## 4. iPhone install (what users do)

1. Open your Render URL in **Safari**.
2. Tap **Share → Add to Home Screen**.
3. It installs like an app: icon, full-screen, offline shell.

## 5. AI (what users do — BYOK)

- Tap ⚙️ in the app → choose OpenAI or Anthropic → paste their own API key.
- The key is stored **only in their browser/device storage** and every AI
  request goes **directly from their device to the provider** — it never
  touches your server. That's the standard trust model used by BYOK apps
  (TypingMind, Chatbot UI, etc.), and you can point skeptical users at the
  network tab to verify it.
- Without a key, the full stats engine still works free.

## Costs & limits recap

| Piece            | Provider        | Free tier limit                  |
|------------------|-----------------|----------------------------------|
| Hosting          | Render          | sleeps when idle; 750 h/month    |
| Database + auth  | Supabase        | 500 MB DB, 50k monthly users     |
| Scheduled crawler| GitHub Actions  | 2,000 min/month (crawler ≈90)    |
| AI               | user's own key  | whatever the user pays their provider |
