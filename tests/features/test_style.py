import math
import pandas as pd
import pytest
from src.features.style import team_style_vector, style_mismatch


def test_style_mismatch_distance_and_xt_diff():
    ts = pd.DataFrame({
        "team": ["Spain", "Spain", "Italy", "Italy"],
        "season": ["2024"] * 4,
        "metric": ["share_pass", "mean_xt", "share_pass", "mean_xt"],
        "value":  [0.7, 0.30, 0.4, 0.10],
        "source": ["vaep"] * 4,
    })
    assert team_style_vector(ts, "Spain", "2024") == {"share_pass": 0.7, "mean_xt": 0.30}
    out = style_mismatch(ts, "Spain", "Italy", "2024")
    # shared {share_pass, mean_xt}: sqrt(0.3^2 + 0.2^2)
    assert abs(out["style_mismatch"] - (0.3 ** 2 + 0.2 ** 2) ** 0.5) < 1e-9
    assert abs(out["xt_diff"] - 0.20) < 1e-9


def test_style_mismatch_no_shared_metrics_is_nan():
    ts = pd.DataFrame({
        "team":   ["Spain", "Italy"],
        "season": ["2024", "2024"],
        "metric": ["share_pass", "share_dribble"],   # disjoint metric sets
        "value":  [0.7, 0.3],
        "source": ["vaep", "vaep"],
    })
    out = style_mismatch(ts, "Spain", "Italy", "2024")
    assert math.isnan(out["style_mismatch"])             # guard, not 0.0 from sum([])**0.5


def test_team_style_vector_rejects_duplicate_metrics():
    ts = pd.DataFrame({
        "team":   ["Spain", "Spain"],
        "season": ["2024", "2024"],
        "metric": ["mean_xt", "mean_xt"],                # same metric, two sources
        "value":  [0.30, 0.25],
        "source": ["vaep", "fbref"],
    })
    with pytest.raises(ValueError):
        team_style_vector(ts, "Spain", "2024")
