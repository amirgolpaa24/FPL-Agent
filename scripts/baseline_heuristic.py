"""
Candidate captain-picking formulas. None of these use AI - they're all
plain arithmetic over the features build_dataset.py already computed, so a
future AI-reasoning layer has a real, honest number to beat.

Every formula follows the same contract: given all player rows for one
gameweek (with form_*/opp_*/fixture_* features already computed, using only
information known before that gameweek), return an `expected_score` per
player, restricted to players who've actually been getting regular minutes.
Only the *scoring rule* differs between formulas.

Add a new candidate by writing a function with this signature and registering
it in FORMULAS at the bottom.
"""

import pandas as pd

MIN_MINUTES_FORM = 45  # avg minutes over the form window to be considered "nailed"


# --- reusable fixture-difficulty building blocks ---------------------------

def _fdr_boost(df: pd.DataFrame) -> pd.Series:
    """Official FPL rating (1=easiest..5=hardest) mapped to +1..-1, neutral at 3."""
    return (3 - df["fixture_difficulty"].fillna(3)) * 0.5


def _proxy_boost(df: pd.DataFrame) -> pd.Series:
    """Fallback/alternative signal: opponent's recent goals-conceded form,
    centered on a roughly average opponent (~1.3 goals conceded/game)."""
    return (df["opp_goals_conceded_form"].fillna(1.3) - 1.3).clip(-1, 1)


def _best_available_fixture_boost(df: pd.DataFrame) -> pd.Series:
    """Official FDR when we have it, otherwise the goals-conceded proxy."""
    if "fixture_difficulty" in df.columns and df["fixture_difficulty"].notna().any():
        return _fdr_boost(df)
    return _proxy_boost(df)


# --- candidate formulas -----------------------------------------------------

def formula_form_fdr(df: pd.DataFrame) -> pd.Series:
    """Current baseline: recent points form + official fixture difficulty
    (falls back to the goals-conceded proxy when FDR isn't available)."""
    return df["form_points"].fillna(0) + _best_available_fixture_boost(df)


def formula_form_proxy_only(df: pd.DataFrame) -> pd.Series:
    """Recent points form + opponent's recent goals-conceded proxy, ignoring
    official FDR even when it's available. Tests whether the "leaky defense"
    signal is worth anything on its own."""
    return df["form_points"].fillna(0) + _proxy_boost(df)


def formula_form_blended_fixture(df: pd.DataFrame) -> pd.Series:
    """Average the official FDR and the goals-conceded proxy 50/50 (when both
    exist) instead of picking one or the other."""
    proxy = _proxy_boost(df)
    if "fixture_difficulty" in df.columns and df["fixture_difficulty"].notna().any():
        blended = 0.5 * _fdr_boost(df).fillna(0) + 0.5 * proxy.fillna(0)
    else:
        blended = proxy
    return df["form_points"].fillna(0) + blended


def formula_ict_fixture(df: pd.DataFrame) -> pd.Series:
    """Use rolling ICT index (influence/creativity/threat - the underlying
    performance metrics FPL itself derives bonus points from) as the form
    signal instead of raw points. Points are noisy game-to-game; ICT is a
    smoother read on who's actually playing well."""
    ict_scaled = df["form_ict"].fillna(0) / 4.0  # rough scale-match to points
    return ict_scaled + _best_available_fixture_boost(df)


def formula_momentum(df: pd.DataFrame) -> pd.Series:
    """Naive real-world strategy: captain whoever just had the best single
    last gameweek, no averaging. Included as a sanity-check baseline - the
    thing most casual managers actually do."""
    return df["last_gw_points"].fillna(0) + _best_available_fixture_boost(df)


def formula_expanding_season_form(df: pd.DataFrame) -> pd.Series:
    """Season-long reliability: average points across every prior gameweek
    this season (not just a short recent window). Rewards proven quality
    over hot streaks; will lag when a player's role/form genuinely changes."""
    return df["season_form_points"].fillna(0) + _best_available_fixture_boost(df)


def formula_form_fdr_home_adjusted(df: pd.DataFrame) -> pd.Series:
    """Current baseline plus a small home-advantage nudge."""
    home_boost = df["was_home"].map({True: 0.3, False: -0.3}).fillna(0)
    return df["form_points"].fillna(0) + _best_available_fixture_boost(df) + home_boost


def _availability_multiplier(df: pd.DataFrame) -> pd.Series:
    """Fraction of a full game we expect the player to feature, in [0.1, 1.0].
    Prefers an explicit availability signal (live chance-of-playing, 0-1) when
    present; otherwise infers it from recent minutes. A player who isn't
    starting can't return points, so this scales the whole score down."""
    if "availability" in df.columns and df["availability"].notna().any():
        avail = df["availability"].fillna(1.0)
    else:
        avail = (df["form_minutes"].fillna(0) / 90.0)
    return avail.clip(lower=0.1, upper=1.0)


def formula_composite(df: pd.DataFrame) -> pd.Series:
    """Multi-factor expected-points model - the main engine.

    Blends the signals that matter for FPL returns and scales the result by
    how likely the player is to actually play:

      scoring rate = 0.50 * season-long form   (proven quality; strongest signal)
                   + 0.30 * recent form         (current heat)
                   + 0.20 * ICT/4               (underlying performance / ability)
      context      + fixture boost (official FDR, easy fixture = +, hard = -)
                   + opponent's leakiness (a small extra nudge)
                   + home advantage
                   + momentum (small bonus if last GW beat the season average)
      expected pts = (scoring rate + context) * availability multiplier
    """
    season = df["season_form_points"].fillna(0)
    recent = df["form_points"].fillna(0)
    ict = df["form_ict"].fillna(0) / 4.0

    # Season-long form is the strongest signal in our backtests, so it leads;
    # recent form and ICT (underlying ability) refine it.
    scoring_rate = 0.65 * season + 0.20 * recent + 0.15 * ict

    fixture = _best_available_fixture_boost(df)
    home = df["was_home"].map({True: 0.15, False: -0.15}).fillna(0) if "was_home" in df.columns else 0

    context = fixture + home
    return (scoring_rate + context) * _availability_multiplier(df)


def formula_fpl_official(df: pd.DataFrame) -> pd.Series:
    """FPL's OWN official expected points for the upcoming gameweek (the
    `ep_next` field from the FPL API). This is not a model we invented and
    weighted - it's the figure FPL publishes for every player, with their
    availability/rotation adjustments already applied. For any player missing
    an official number (e.g. historical backtests, where the API field doesn't
    exist), it falls back to the composite model so nothing is left unscored."""
    comp = formula_composite(df)
    if "fpl_ep_next" in df.columns and df["fpl_ep_next"].notna().any():
        official = pd.to_numeric(df["fpl_ep_next"], errors="coerce").fillna(comp)
        # Tiny composite-based tiebreaker: too small to change the displayed
        # 2-decimal xP, but breaks ties sensibly when FPL's figure is flat
        # (e.g. pre-season, when many players share the same ep_next).
        return official + comp * 1e-4
    return comp


FORMULAS = {
    "fpl_official": formula_fpl_official,
    "composite": formula_composite,
    "form_fdr": formula_form_fdr,
    "form_proxy_only": formula_form_proxy_only,
    "form_blended_fixture": formula_form_blended_fixture,
    "ict_fixture": formula_ict_fixture,
    "momentum_last_gw": formula_momentum,
    "expanding_season_form": formula_expanding_season_form,
    "form_fdr_home_adjusted": formula_form_fdr_home_adjusted,
}

DEFAULT_FORMULA = "composite"


def score_gameweek(gw_df: pd.DataFrame, formula: str = DEFAULT_FORMULA) -> pd.DataFrame:
    """Given all player rows for one gameweek, return the same rows with an
    `expected_score` column, sorted best-first, restricted to eligible
    (regularly-playing) players."""
    if formula not in FORMULAS:
        raise ValueError(f"Unknown formula '{formula}'. Options: {list(FORMULAS)}")

    df = gw_df.copy()
    eligible = df["form_minutes"].fillna(0) >= MIN_MINUTES_FORM
    df["expected_score"] = FORMULAS[formula](df)
    df = df[eligible].sort_values("expected_score", ascending=False)
    return df


def pick_captain(gw_df: pd.DataFrame, formula: str = DEFAULT_FORMULA) -> pd.Series | None:
    scored = score_gameweek(gw_df, formula=formula)
    if scored.empty:
        return None
    return scored.iloc[0]


def pick_top_n(gw_df: pd.DataFrame, n: int = 15, formula: str = DEFAULT_FORMULA) -> pd.DataFrame:
    return score_gameweek(gw_df, formula=formula).head(n)
