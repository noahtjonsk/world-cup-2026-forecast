import pandas as pd
from src.report.match_view import match_card, top_drivers

PRED = {"home_team": "A", "away_team": "B", "p_home": 0.5, "p_draw": 0.3, "p_away": 0.2,
        "exp_goals_home": 1.4, "exp_goals_away": 1.1}


def test_match_card_grid_normalised_and_top_scoreline():
    card = match_card(PRED, rho=0.0, max_goals=10, top_scorelines=3)
    assert card["home_team"] == "A" and card["p_home"] == 0.5
    assert abs(card["scoreline_grid"].sum() - 1.0) < 1e-9        # score_matrix renormalises
    assert card["scoreline_grid"].shape == (11, 11)
    assert card["top_scorelines"][0][0] == (1, 1)               # mode of Pois(1.4) x Pois(1.1)
    assert len(card["top_scorelines"]) == 3
    assert card["top_scorelines"][0][1] >= card["top_scorelines"][1][1]   # sorted desc


def test_top_drivers_ranks_by_importance_skips_absent():
    feature_row = {"elo_diff": 50.0, "form_diff": 0.3, "xi_quality_diff": 0.1, "style_mismatch": -0.2}
    importances = pd.DataFrame([
        {"feature": "missing_feat", "importance": 99.0},   # highest, but not in the row -> skipped
        {"feature": "elo_diff", "importance": 40.0},
        {"feature": "xi_quality_diff", "importance": 25.0},
        {"feature": "form_diff", "importance": 20.0},
    ])
    out = top_drivers(feature_row, importances, k=2)
    assert list(out.columns) == ["feature", "importance", "value"]
    assert list(out["feature"]) == ["elo_diff", "xi_quality_diff"]
    assert out.iloc[0]["value"] == 50.0
