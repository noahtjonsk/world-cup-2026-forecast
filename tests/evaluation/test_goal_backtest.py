import numpy as np
from src.evaluation.goal_backtest import wdl_probs_for_matches, goalmodel_walkforward

def test_wdl_probs_for_matches_assembly_and_order():
    # A is much stronger than B (gap > home_adv) so A wins home AND away
    params = {"attack": {"A": 0.6, "B": -0.6}, "defence": {"A": -0.4, "B": 0.4},
              "home_adv": 0.25, "rho": 0.0}
    P = wdl_probs_for_matches(params, ["A", "B"], ["B", "A"], rho=0.0, max_goals=10)
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    # A stronger AND home in row0 -> p_home > p_away; A stronger away in row1 -> p_away > p_home
    assert P[0, 0] > P[0, 2] and P[1, 2] > P[1, 0]

def test_backtest_importable_without_scipy():
    assert callable(goalmodel_walkforward)

def test_squad_coef_walkforward_is_callable():
    from src.evaluation.goal_backtest import squad_coef_walkforward
    assert callable(squad_coef_walkforward)


def test_elo_anchor_walkforward_is_callable():
    from src.evaluation.goal_backtest import elo_anchor_walkforward
    assert callable(elo_anchor_walkforward)


def test_squad_xconf_walkforward_is_callable():
    from src.evaluation.goal_backtest import squad_xconf_walkforward
    assert callable(squad_xconf_walkforward)
