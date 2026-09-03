import numpy as np
from src.models.elo_prior import elo_prior_net

def test_standardize_center_scale_and_align():
    elo = {"A": 1600.0, "B": 1500.0, "C": 1400.0}     # mean 1500, sd(ddof=0)=81.6497
    teams = ["A", "B", "C"]
    beta = 0.35
    out = elo_prior_net(elo, teams, beta=beta)
    z = np.array([100.0, 0.0, -100.0]) / np.std([1600, 1500, 1400])  # ddof=0
    assert np.allclose(out, beta * z)
    assert abs(out.mean()) < 1e-12                      # mean-zero like atk

def test_missing_team_maps_to_zero_neutral():
    out = elo_prior_net({"A": 1600.0, "B": 1400.0}, ["A", "B", "C"], beta=0.35)
    assert out[2] == 0.0                                # C unseen -> neutral
    # A,B standardized over the TWO present teams (mean 1500, sd 100): z=[1,-1]
    assert np.allclose(out[:2], 0.35 * np.array([1.0, -1.0]))

def test_zero_variance_returns_zeros_no_div0():
    out = elo_prior_net({"A": 1500.0, "B": 1500.0}, ["A", "B"], beta=0.35)
    assert np.allclose(out, 0.0)

def test_empty_or_all_missing_returns_zeros():
    assert np.allclose(elo_prior_net({}, ["A", "B"], beta=0.35), 0.0)
