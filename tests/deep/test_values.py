from src.schema import CANON_ACTION_COLS, CANON_TEAM_STYLE_COLS

def test_deep_schemas_present():
    assert CANON_TEAM_STYLE_COLS == ["team", "season", "metric", "value", "source"]
    assert CANON_ACTION_COLS[0] == "match_id"
    assert {"vaep_value", "offensive_value", "defensive_value", "xt_value", "action_type"} <= set(CANON_ACTION_COLS)

import pandas as pd
from src.schema import CANON_PLAYER_STAT_COLS
from src.deep.values import aggregate_player_values

def test_aggregate_per90_filters_low_minutes():
    actions = pd.DataFrame({
        "player": ["Rodri", "Rodri", "Foden"],
        "team":   ["Man City", "Man City", "Man City"],
        "vaep_value":       [0.6, 0.6, 0.5],
        "offensive_value":  [0.4, 0.4, 0.5],
        "defensive_value":  [0.2, 0.2, 0.0],
        "xt_value":         [0.3, 0.1, 0.2],
    })
    players = pd.DataFrame({"player": ["Rodri", "Foden"], "minutes_played": [240, 90]})
    out = aggregate_player_values(actions, players, season="2024-2025", min_minutes=180)
    assert list(out.columns) == CANON_PLAYER_STAT_COLS
    assert set(out["player"]) == {"Rodri"}                       # Foden filtered (90 <= 180)
    assert set(out["metric"]) == {"vaep_p90", "vaep_off_p90", "vaep_def_p90", "xt_p90"}
    v = out.loc[out["metric"] == "vaep_p90", "value"].iloc[0]
    assert abs(v - (1.2 * 90 / 240)) < 1e-9                      # 0.45

from src.deep.values import decompose_vaep_by_action_type

def test_decompose_vaep_by_action_type():
    actions = pd.DataFrame({
        "player": ["Rodri", "Rodri", "Rodri", "Foden"],
        "team":   ["Man City"] * 4,
        "action_type": ["pass", "pass", "carry", "pass"],
        "vaep_value": [0.6, 0.6, 0.3, 0.9],
    })
    players = pd.DataFrame({"player": ["Rodri", "Foden"], "minutes_played": [240, 90]})
    out = decompose_vaep_by_action_type(actions, players, season="2024-2025", min_minutes=180)
    assert list(out.columns) == CANON_PLAYER_STAT_COLS
    assert {"vaep_pass_p90", "vaep_carry_p90"} <= set(out["metric"])
    pass_val = out.loc[out["metric"] == "vaep_pass_p90", "value"].iloc[0]
    assert abs(pass_val - (1.2 * 90 / 240)) < 1e-9               # 0.45
    assert "Foden" not in set(out["player"])                    # filtered

from src.deep.values import compute_team_style

def test_compute_team_style_shares_and_mean_xt():
    actions = pd.DataFrame({
        "team": ["Spain"] * 3,
        "action_type": ["pass", "pass", "carry"],
        "xt_value": [0.2, 0.0, 0.4],
    })
    out = compute_team_style(actions, season="2024")
    assert list(out.columns) == CANON_TEAM_STYLE_COLS
    metrics = dict(zip(out["metric"], out["value"]))
    assert abs(metrics["share_pass"] - 2 / 3) < 1e-9
    assert abs(metrics["share_carry"] - 1 / 3) < 1e-9
    assert abs(metrics["mean_xt"] - 0.2) < 1e-9                  # (0.2+0.0+0.4)/3

from src.deep.values import assemble_deep_tables

def test_assemble_returns_player_values_and_team_style():
    actions = pd.DataFrame({
        "player": ["Rodri", "Rodri"], "team": ["Spain", "Spain"],
        "action_type": ["pass", "carry"],
        "vaep_value": [0.6, 0.4], "offensive_value": [0.4, 0.3],
        "defensive_value": [0.2, 0.1], "xt_value": [0.3, 0.2],
    })
    players = pd.DataFrame({"player": ["Rodri"], "minutes_played": [240]})
    tables = assemble_deep_tables(actions, players, season="2024", min_minutes=180)
    assert set(tables) == {"player_values", "team_style"}
    assert list(tables["player_values"].columns) == CANON_PLAYER_STAT_COLS
    assert list(tables["team_style"].columns) == CANON_TEAM_STYLE_COLS
    # player_values = per-90 totals (4) + per-type (pass, carry) = 6 rows
    assert len(tables["player_values"]) == 6
