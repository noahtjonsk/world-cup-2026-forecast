import pandas as pd
from src.features.squad import squad_features

def test_squad_features_xi_bench_coverage():
    lineup = pd.DataFrame({
        "team": ["Spain"] * 3,
        "player": ["A", "B", "C"],
        "is_starter": [True, True, False],
    })
    quality = pd.DataFrame({
        "player": ["A", "B", "C"],
        "role":   ["ST", "CB", "ST"],
        "quality": [1.0, 0.6, 0.2],
    })
    s = squad_features(lineup, quality)["Spain"]
    assert abs(s["xi_quality"] - 0.8) < 1e-9              # mean(1.0, 0.6)
    assert abs(s["bench_dropoff"] - (0.8 - 0.2)) < 1e-9   # xi - bench mean
    assert abs(s["role_coverage"] - (1.6 / 8)) < 1e-9     # (ST best 1.0 + CB best 0.6 + 6*0) / 8


def test_squad_features_no_bench_gives_nan_dropoff():
    lineup = pd.DataFrame({"team": ["Spain", "Spain"], "player": ["A", "B"], "is_starter": [True, True]})
    quality = pd.DataFrame({"player": ["A", "B"], "role": ["ST", "CB"], "quality": [1.0, 0.6]})
    s = squad_features(lineup, quality)["Spain"]
    assert s["bench_dropoff"] != s["bench_dropoff"]        # NaN (no bench)


def test_squad_features_unscored_role_counts_as_zero():
    # B has a valid role (CB) but NaN quality -> CB's best is NaN; role_coverage must
    # treat that role as 0, not let the NaN contaminate the whole feature.
    lineup = pd.DataFrame({"team": ["Spain", "Spain"], "player": ["A", "B"], "is_starter": [True, True]})
    quality = pd.DataFrame({"player": ["A", "B"], "role": ["ST", "CB"], "quality": [1.0, float("nan")]})
    s = squad_features(lineup, quality)["Spain"]
    assert s["role_coverage"] == s["role_coverage"]        # not NaN
    assert abs(s["role_coverage"] - (1.0 / 8)) < 1e-9      # ST=1.0, CB(unscored)->0, others 0
