import pandas as pd
from src.states.predict import upcoming_fixtures

def test_upcoming_fixtures_excludes_played_and_past():
    fx = pd.DataFrame([
        {"fixture_id": 1, "date": "2026-06-11", "status": "FT", "home_team": "A", "away_team": "B"},
        {"fixture_id": 2, "date": "2026-06-20", "status": "NS", "home_team": "C", "away_team": "D"},
        {"fixture_id": 3, "date": "2026-06-05", "status": "NS", "home_team": "E", "away_team": "F"},
    ])
    up = upcoming_fixtures(fx, as_of="2026-06-12")
    assert list(up["fixture_id"]) == [2]            # 1 already played (FT), 3 in the past

from src.states.predict import assemble_predictions
from src.schema import CANON_PREDICTION_COLS

def test_assemble_predictions_cols_alignment_and_keys():
    fx = pd.DataFrame([
        {"fixture_id": 2, "date": "2026-06-20", "home_team": "C", "away_team": "D"},
        {"fixture_id": 5, "date": "2026-06-21", "home_team": "E", "away_team": "F"},
    ])
    out = assemble_predictions(fx, wdl_probs=[[0.5, 0.3, 0.2], [0.25, 0.25, 0.5]],
                               exp_goals=[[1.6, 1.1], [0.9, 1.7]], snapshot_date="2026-06-12")
    assert list(out.columns) == CANON_PREDICTION_COLS
    r0 = out.iloc[0]
    assert abs(r0["p_home"] - 0.5) < 1e-9 and abs(r0["p_away"] - 0.2) < 1e-9
    assert abs(r0["exp_goals_home"] - 1.6) < 1e-9 and abs(r0["exp_goals_away"] - 1.1) < 1e-9
    assert r0["home_team"] == "C" and r0["away_team"] == "D"
    assert out["match_id"].notna().all() and out.iloc[1]["match_id"] != r0["match_id"]
    assert (out["snapshot_date"] == pd.Timestamp("2026-06-12")).all()
