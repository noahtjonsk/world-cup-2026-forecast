import pandas as pd
from src.report.tournament_view import progression_table, qualification_view


def _results():
    return pd.DataFrame([
        {"tournament": "T", "team": "A", "round": "F", "prob": 0.60, "ci_low": 0.5, "ci_high": 0.7},
        {"tournament": "T", "team": "A", "round": "W", "prob": 0.40, "ci_low": 0.3, "ci_high": 0.5},
        {"tournament": "T", "team": "B", "round": "F", "prob": 0.40, "ci_low": 0.3, "ci_high": 0.5},
        {"tournament": "T", "team": "B", "round": "W", "prob": 0.10, "ci_low": 0.05, "ci_high": 0.2},
        {"tournament": "X", "team": "Z", "round": "W", "prob": 0.99, "ci_low": 0.9, "ci_high": 1.0},
    ])


def test_progression_table_wide_ordered_by_title():
    out = progression_table(_results(), tournament="T", round_order=["F", "W"])
    assert list(out.columns) == ["team", "F", "W"]
    assert list(out["team"]) == ["A", "B"]                  # sorted by P(win) desc; tournament X excluded
    idx = out.set_index("team")
    assert idx.loc["A", "W"] == 0.40 and idx.loc["B", "F"] == 0.40


def test_qualification_view_round_ordered_and_filtered():
    v = qualification_view(_results(), tournament="T", teams=["A"], round_order=["F", "W"])
    assert list(v.columns) == ["team", "round", "prob", "ci_low", "ci_high"]
    assert list(v["round"]) == ["F", "W"]                   # ordered categorical, sorted
    assert list(v["prob"]) == [0.60, 0.40]
    assert set(v["team"]) == {"A"}                          # filtered
