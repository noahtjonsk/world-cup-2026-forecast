from src.models.elo_baseline import elo_expected_score

def _gd_multiplier(goal_diff):
    """eloratings.net goal-difference index G: 1 for |gd|<=1, 1.5 for |gd|==2,
    (11+|gd|)/8 for |gd|>=3, so blowouts move ratings more, with diminishing returns."""
    g = abs(int(goal_diff))
    if g <= 1:
        return 1.0
    if g == 2:
        return 1.5
    return (11 + g) / 8.0

def elo_update_one(elo_home, elo_away, home_score, away_score, k=60.0,
                   home_adv=0.0, neutral=True):
    """One eloratings.net-style symmetric Elo update from the home team's view.
    Returns (new_elo_home, new_elo_away). delta = K * G * (W - We), W = 1/0.5/0,
    We = Elo win-expectancy (home_adv zeroed when neutral, WC matches are neutral).
    Zero-sum: home gains delta, away loses it (total Elo conserved)."""
    we = elo_expected_score(elo_home - elo_away, home_adv=home_adv, neutral=neutral)
    w = 1.0 if home_score > away_score else (0.0 if home_score < away_score else 0.5)
    delta = k * _gd_multiplier(home_score - away_score) * (w - we)
    return elo_home + delta, elo_away - delta

import pandas as pd
from src.schema import CANON_RATING_COLS

def seed_from_ratings(ratings, cutoff):
    """{team: most-recent Elo strictly before `cutoff`}, the prior to replay from
    (e.g. the latest eloratings value before the 2026 tournament)."""
    df = ratings.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] < pd.Timestamp(cutoff)].sort_values("date", kind="mergesort")
    return {t: float(sub["elo"].iloc[-1]) for t, sub in df.groupby("team")}

def recompute_elo(matches, seed_ratings, k_by_competition=None, default_k=60.0,
                  home_adv=0.0, source="live_elo", since=None):
    """Replay `matches` in date order from `seed_ratings`, emitting one dated
    post-match rating row per team per match (CANON_RATING_COLS). Idempotent &
    deterministic: same matches + same seed -> identical output (re-runs never
    double-count). Unknown teams seed at 1500. `k_by_competition` maps competition
    -> K (match importance); `default_k` otherwise. WC matches are neutral.

    `since`: replay only matches on/after this date. The seed is the rating AS OF
    a cutoff, so pass that cutoff here, replaying pre-cutoff matches re-applies
    history already priced into the seed (the bug that corrupted elo_diff by
    +110 points on the Match page)."""
    k_by_competition = k_by_competition or {}
    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", kind="mergesort")
    elo = dict(seed_ratings)
    rows = []
    for _, m in df.iterrows():
        h, a = m["home_team"], m["away_team"]
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        k = k_by_competition.get(str(m["competition"]), default_k)
        neutral = True if pd.isna(m.get("neutral")) else bool(m["neutral"])
        nh, na = elo_update_one(eh, ea, int(m["home_score"]), int(m["away_score"]),
                                k=k, home_adv=home_adv, neutral=neutral)
        elo[h], elo[a] = nh, na
        rows.append({"date": m["date"], "team": h, "elo": nh, "source": source})
        rows.append({"date": m["date"], "team": a, "elo": na, "source": source})
    if not rows:
        return pd.DataFrame(columns=CANON_RATING_COLS)
    return pd.DataFrame(rows)[CANON_RATING_COLS]
