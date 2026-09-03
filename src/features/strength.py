# src/features/strength.py
import pandas as pd
from src.utils.dates import asof_window

def latest_elo_asof(ratings, team, kickoff, months=24):
    """Most recent Elo for `team` strictly before kickoff AND within the window."""
    sub = ratings[ratings["team"] == team]
    sub = asof_window(sub, kickoff, months=months)
    if sub.empty:
        return float("nan")
    # asof_window leaves the date column dtype untouched, so coerce before sorting
    # (mirrors recent_form), string dates would otherwise sort lexicographically.
    sub = sub.assign(date=pd.to_datetime(sub["date"]))
    return float(sub.sort_values("date")["elo"].iloc[-1])

def recent_form(matches, team, kickoff, months=24, half_life_days=365):
    """Recency-weighted points-per-game (3/1/0) from results strictly before kickoff
    and within the window. Exponential half-life weighting by recency."""
    m = matches[(matches["home_team"] == team) | (matches["away_team"] == team)]
    m = asof_window(m, kickoff, months=months)  # already returns a copy
    if m.empty:
        return float("nan")
    m["date"] = pd.to_datetime(m["date"])
    is_home = m["home_team"] == team
    gf = m["home_score"].where(is_home, m["away_score"])
    ga = m["away_score"].where(is_home, m["home_score"])
    pts = pd.Series(1, index=m.index).mask(gf > ga, 3).mask(gf < ga, 0)
    age_days = (pd.Timestamp(kickoff) - m["date"]).dt.days
    w = 0.5 ** (age_days / half_life_days)
    return float((pts * w).sum() / w.sum())

def strength_features(matches, ratings, match_row, months=24):
    """Elo + form features for one match row -> dict.

    Any feature whose inputs are missing within the window is NaN (and the
    corresponding *_diff is NaN), so downstream model code must tolerate NaN."""
    home, away, kickoff = match_row["home_team"], match_row["away_team"], match_row["date"]
    eh = latest_elo_asof(ratings, home, kickoff, months)
    ea = latest_elo_asof(ratings, away, kickoff, months)
    fh = recent_form(matches, home, kickoff, months)
    fa = recent_form(matches, away, kickoff, months)
    return {
        "elo_home": eh, "elo_away": ea, "elo_diff": eh - ea,
        "form_home": fh, "form_away": fa, "form_diff": fh - fa,
    }
