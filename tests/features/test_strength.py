# tests/features/test_strength.py
import pandas as pd
from src.features.strength import strength_features

def test_strength_features_elo_form_leakage_safe():
    ratings = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2027-01-01"]),
        "team": ["Spain", "Italy", "Spain"],
        "elo":  [2050.0, 1900.0, 9999.0],     # last row is POST-kickoff -> must be ignored
        "source": ["eloratings"] * 3,
    })
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2026-05-01", "2026-04-01", "2026-03-01"]),
        "home_team": ["Spain", "Spain", "Italy"],
        "away_team": ["Italy", "France", "Spain"],
        "home_score": [2, 0, 1],
        "away_score": [0, 0, 1],
    })
    row = {"home_team": "Spain", "away_team": "Italy", "date": pd.Timestamp("2026-06-01")}
    f = strength_features(matches, ratings, row, months=24)
    assert f["elo_home"] == 2050.0 and f["elo_away"] == 1900.0
    assert f["elo_diff"] == 150.0
    # Spain pool: win(3)+draw(1)+draw(1); Italy pool: loss(0)+draw(1) -> home form > away form
    assert f["form_home"] > f["form_away"]
