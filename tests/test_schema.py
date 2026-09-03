from src.schema import (
    CANON_RATING_COLS, CANON_PLAYER_STAT_COLS, CANON_FIXTURE_COLS,
    CANON_LINEUP_COLS, CANON_INJURY_COLS,
)

def test_new_schemas_present_and_shaped():
    assert CANON_RATING_COLS == ["date", "team", "elo", "source"]
    assert CANON_PLAYER_STAT_COLS == ["player", "team", "season", "position", "metric", "value", "source"]
    assert CANON_FIXTURE_COLS[0] == "fixture_id" and "neutral" in CANON_FIXTURE_COLS
    assert CANON_LINEUP_COLS == ["fixture_id", "team", "player", "position", "is_starter", "formation", "source"]
    assert CANON_INJURY_COLS == ["date", "team", "player", "reason", "status", "source"]

def test_sim_schema_cols():
    from src.schema import CANON_GROUP_COLS, CANON_SIM_RESULT_COLS
    assert CANON_GROUP_COLS == ["tournament", "group", "team"]
    assert CANON_SIM_RESULT_COLS == ["tournament", "team", "round", "prob", "ci_low", "ci_high"]

def test_prediction_schema_cols():
    from src.schema import CANON_PREDICTION_COLS
    assert CANON_PREDICTION_COLS == [
        "match_id", "date", "home_team", "away_team", "snapshot_date",
        "p_home", "p_draw", "p_away", "exp_goals_home", "exp_goals_away", "source",
    ]
