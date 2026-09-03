import math
import numpy as np
from src.simulation.match import match_lambdas, match_probs, simulate_match, knockout_winner

PARAMS = {"attack": {"A": 0.5, "B": 0.1}, "defence": {"A": 0.2, "B": -0.1},
          "home_adv": 0.3, "rho": 0.0}


def test_match_lambdas_host_gets_home_adv_else_neutral():
    lam_h, lam_a = match_lambdas(PARAMS, "A", "B", hosts=["A"])     # A hosts -> home_adv
    assert abs(lam_h - math.exp(0.5 - (-0.1) + 0.3)) < 1e-9
    assert abs(lam_a - math.exp(0.1 - 0.2)) < 1e-9
    lam_h_n, _ = match_lambdas(PARAMS, "A", "B", hosts=[])          # neutral -> no home_adv
    assert abs(lam_h_n - math.exp(0.5 - (-0.1))) < 1e-9


def test_match_probs_symmetric_equal_strength():
    ph, pdr, pa = match_probs(1.3, 1.3, rho=0.0)
    assert abs(ph + pdr + pa - 1.0) < 1e-9 and abs(ph - pa) < 1e-9


def test_simulate_match_strong_favorite_wins_most():
    rng = np.random.default_rng(42)
    wins = sum(simulate_match(4.0, 0.2, rng)[2] == "home" for _ in range(500))
    assert wins / 500 > 0.9


def test_group_match_can_draw_but_knockout_cannot():
    rng = np.random.default_rng(1)
    group = [simulate_match(1.1, 1.1, rng, knockout=False)[2] for _ in range(300)]
    knock = [simulate_match(1.1, 1.1, rng, knockout=True)[2] for _ in range(300)]
    assert "draw" in group
    assert "draw" not in knock and set(knock) <= {"home", "away"}


def test_knockout_winner_extra_time_favors_the_stronger_side():
    # A drawn knockout goes to a short extra time at et_factor * lam, THEN a 50/50
    # shootout, so the favorite must win clearly more than a coin flip overall.
    rng = np.random.default_rng(7)
    n = 4000
    home = sum(knockout_winner(3.0, 0.3, rng) == "home" for _ in range(n))
    assert home / n > 0.60                       # was exactly 0.50 under the pure coin flip


def test_knockout_winner_even_match_is_near_fair():
    rng = np.random.default_rng(11)
    n = 4000
    home = sum(knockout_winner(1.2, 1.2, rng) == "home" for _ in range(n))
    assert 0.46 < home / n < 0.54                # symmetric lambdas -> ~50/50


def test_unseen_team_regresses_to_league_average():
    """Teams not in the fitted params get atk=0, def=0 (league-average)."""
    lam_h, lam_a = match_lambdas(PARAMS, "A", "UNSEEN", hosts=[])
    # atk["A"]=0.5, dfc["UNSEEN"] defaults to 0, home_adv=0
    assert abs(lam_h - math.exp(0.5 - 0.0)) < 1e-9
    # atk["UNSEEN"] defaults to 0, dfc["A"]=0.2
    assert abs(lam_a - math.exp(0.0 - 0.2)) < 1e-9
