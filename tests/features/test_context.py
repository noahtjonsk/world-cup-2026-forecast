import pandas as pd
from src.features.context import stage_code, rest_days, context_features

def test_stage_code_and_context():
    assert stage_code("Group") == 0
    assert stage_code("Round of 16") == 2
    assert stage_code("Final") == 5
    assert stage_code(None) == 0

    matches = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01", "2026-05-25"]),
        "home_team": ["USA", "Brazil"],
        "away_team": ["Brazil", "USA"],
        "home_score": [1, 2],
        "away_score": [1, 0],
    })
    assert rest_days(matches, "USA", "2026-06-10") == 9.0   # 06-10 minus most-recent 06-01

    row = {"home_team": "USA", "away_team": "Brazil", "date": pd.Timestamp("2026-06-10"),
           "stage": "Round of 16", "neutral": False}
    out = context_features(matches, row, host_teams=("USA", "Canada", "Mexico"))
    assert out["rest_home"] == 9.0 and out["rest_away"] == 9.0 and out["rest_diff"] == 0.0
    assert out["stage_code"] == 2
    assert out["neutral"] == 0
    assert out["host_home"] == 1 and out["host_away"] == 0


def test_rest_days_no_prior_match_is_nan():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01"]),
        "home_team": ["USA"], "away_team": ["Brazil"],
        "home_score": [1], "away_score": [1],
    })
    assert pd.isna(rest_days(matches, "Argentina", "2026-06-10"))   # no prior match -> NaN, no crash


def test_context_features_missing_neutral_defaults_to_zero():
    matches = pd.DataFrame({
        "date": pd.to_datetime(["2026-06-01"]),
        "home_team": ["USA"], "away_team": ["Brazil"],
        "home_score": [1], "away_score": [1],
    })
    row = {"home_team": "USA", "away_team": "Brazil", "date": pd.Timestamp("2026-06-10"),
           "stage": "Group", "neutral": float("nan")}    # NaN must not crash or flag neutral
    out = context_features(matches, row)
    assert out["neutral"] == 0
