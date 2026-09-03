import pandas as pd
from src.simulation.groups import parse_groups, groups_to_dict
from src.schema import CANON_GROUP_COLS


def _fixtures():
    return pd.DataFrame([
        {"stage": "Group A", "home_team": "USA", "away_team": "Wales"},
        {"stage": "Group A", "home_team": "England", "away_team": "Iran"},
        {"stage": "Group B", "home_team": "Spain", "away_team": "Japan"},
        {"stage": "Round of 16", "home_team": "USA", "away_team": "England"},
    ])


def test_parse_groups_from_stage_labels():
    g = parse_groups(_fixtures(), tournament="2026")
    assert list(g.columns) == CANON_GROUP_COLS
    assert set(g[g["group"] == "A"]["team"]) == {"USA", "Wales", "England", "Iran"}
    assert set(g["group"]) == {"A", "B"}                     # knockout fixtures ignored


def test_groups_to_dict():
    d = groups_to_dict(parse_groups(_fixtures(), tournament="2026"))
    assert set(d["B"]) == {"Spain", "Japan"} and set(d.keys()) == {"A", "B"}
