import pandas as pd
from src.report.team_view import style_profile, strength_summary


def test_style_profile_latest_season_only():
    ts = pd.DataFrame([
        {"team": "A", "season": "2024", "metric": "pass_share", "value": 0.40, "source": "sb"},
        {"team": "A", "season": "2024", "metric": "press_share", "value": 0.10, "source": "sb"},
        {"team": "A", "season": "2022", "metric": "pass_share", "value": 0.90, "source": "sb"},
        {"team": "B", "season": "2024", "metric": "pass_share", "value": 0.50, "source": "sb"},
    ])
    out = style_profile(ts, "A")
    assert list(out.columns) == ["metric", "value"]
    assert list(out["metric"]) == ["pass_share", "press_share"]
    assert out.set_index("metric").loc["pass_share", "value"] == 0.40   # 2024, not the 2022 0.90


def test_strength_summary_latest_elo_and_features():
    ratings = pd.DataFrame([
        {"date": "2026-05-01", "team": "A", "elo": 1700.0, "source": "e"},
        {"date": "2026-06-01", "team": "A", "elo": 1750.0, "source": "e"},
    ])
    feats = pd.DataFrame([
        {"date": "2026-06-05", "home_team": "A", "away_team": "B",
         "xi_quality_home": 0.8, "xi_quality_away": 0.6,
         "role_coverage_home": 0.9, "role_coverage_away": 0.7},
    ])
    s = strength_summary(ratings, feats, "A")
    assert s["elo"] == 1750.0 and s["xi_quality"] == 0.8 and s["role_coverage"] == 0.9
