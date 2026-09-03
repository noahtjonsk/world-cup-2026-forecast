# tests/models/test_elo_baseline.py
import numpy as np
from src.models.elo_baseline import (
    elo_expected_score, elo_wdl_probs, predict_proba, fit_draw_base,
)

def test_expected_score_neutral_and_home():
    assert abs(elo_expected_score(0, neutral=True) - 0.5) < 1e-12
    assert elo_expected_score(0, home_adv=65.0, neutral=False) > 0.5      # home edge

def test_wdl_probs_even_match():
    p = elo_wdl_probs(0, neutral=True, draw_base=0.30)
    assert abs(p["H"] + p["D"] + p["A"] - 1.0) < 1e-12
    assert abs(p["D"] - 0.30) < 1e-12                                     # base at We=0.5
    assert abs(p["H"] - 0.35) < 1e-12 and abs(p["A"] - 0.35) < 1e-12

def test_wdl_probs_home_favoured_when_not_neutral():
    p = elo_wdl_probs(0, home_adv=65.0, neutral=False)
    assert p["H"] > p["A"] and abs(sum(p.values()) - 1.0) < 1e-12

def test_predict_proba_vectorized_shape_order():
    P = predict_proba([0, 200], [True, False])                           # neutral, then home
    assert P.shape == (2, 3)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert P[1, 0] > P[1, 2]                                             # row1 home strongly favoured

def test_fit_draw_base_picks_high_when_all_draws():
    # all-draw outcomes at even strength: log-loss is minimised by the largest draw mass
    out = fit_draw_base(np.zeros(30), np.ones(30, dtype=bool), ["D"] * 30,
                        grid=np.linspace(0.05, 0.45, 9))
    assert out == 0.45
