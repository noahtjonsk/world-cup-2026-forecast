CANON_MATCH_COLS = [
    "match_id", "date", "competition", "season",
    "home_team", "away_team", "home_score", "away_score",
    "stage", "neutral", "source",
]

CANON_RATING_COLS = ["date", "team", "elo", "source"]

CANON_PLAYER_STAT_COLS = ["player", "team", "season", "position", "metric", "value", "source"]

CANON_FIXTURE_COLS = [
    "fixture_id", "date", "competition", "season", "stage",
    "home_team", "away_team", "status", "venue", "neutral", "source",
]

CANON_LINEUP_COLS = ["fixture_id", "team", "player", "position", "is_starter", "formation", "source"]

CANON_INJURY_COLS = ["date", "team", "player", "reason", "status", "source"]

CANON_ACTION_COLS = [
    "match_id", "date", "team", "player", "action_type", "result",
    "start_x", "start_y", "end_x", "end_y",
    "vaep_value", "offensive_value", "defensive_value", "xt_value", "source",
]
CANON_TEAM_STYLE_COLS = ["team", "season", "metric", "value", "source"]

CANON_FEATURE_COLS = [
    "match_id", "date", "home_team", "away_team", "snapshot_date",
    "elo_home", "elo_away", "elo_diff",
    "form_home", "form_away", "form_diff",
    "xi_quality_home", "xi_quality_away", "xi_quality_diff",
    "bench_dropoff_home", "bench_dropoff_away",
    "role_coverage_home", "role_coverage_away",
    "style_mismatch", "xt_diff",
    "rest_home", "rest_away", "rest_diff",
    "stage_code", "neutral", "host_home", "host_away",
    "result",
]

CANON_GROUP_COLS = ["tournament", "group", "team"]

CANON_SIM_RESULT_COLS = ["tournament", "team", "round", "prob", "ci_low", "ci_high"]

# Knockout-bracket occupancy counts from the Monte-Carlo: how often each (home, away)
# pairing occurred in each bracket slot, and how often the home side won it.
CANON_BRACKET_COLS = ["tournament", "round", "match_idx", "home_team", "away_team", "n", "home_wins"]

CANON_PREDICTION_COLS = [
    "match_id", "date", "home_team", "away_team", "snapshot_date",
    "p_home", "p_draw", "p_away", "exp_goals_home", "exp_goals_away", "source",
]
