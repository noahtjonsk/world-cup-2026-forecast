# tests/models/test_dixon_coles.py
import math
from src.models.dixon_coles import team_lambdas, fit_dixon_coles

def test_team_lambdas_from_params():
    params = {"attack": {"A": 0.5, "B": 0.1},
              "defence": {"A": 0.2, "B": -0.1},
              "home_adv": 0.3}
    lam_h, lam_a = team_lambdas(params, "A", "B")
    # lam_h = exp(atk_A - dfc_B + ha) = exp(0.5 - (-0.1) + 0.3) = exp(0.9)
    # lam_a = exp(atk_B - dfc_A)      = exp(0.1 - 0.2)          = exp(-0.1)
    assert abs(lam_h - math.exp(0.9)) < 1e-9
    assert abs(lam_a - math.exp(-0.1)) < 1e-9

def test_fit_dixon_coles_is_a_thin_wrapper():
    # scipy is an optional runtime dep -> not fixture-run here, only existence-checked
    assert callable(fit_dixon_coles)

def test_fit_warns_when_optimizer_does_not_converge():
    pytest = __import__("pytest")
    pytest.importorskip("scipy")          # optional dep; suite stays green without it
    import warnings
    import pandas as pd
    matches = pd.DataFrame([
        {"date": "2024-01-01", "competition": "Friendly", "home_team": "A", "away_team": "B",
         "home_score": 2, "away_score": 0},
        {"date": "2024-02-01", "competition": "Friendly", "home_team": "B", "away_team": "A",
         "home_score": 1, "away_score": 1},
        {"date": "2024-03-01", "competition": "Friendly", "home_team": "A", "away_team": "B",
         "home_score": 3, "away_score": 1},
    ])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_dixon_coles(matches, max_iter=1)          # forced early stop -> must warn
    assert any("converge" in str(w.message).lower() for w in caught)


def test_fit_signature_is_backward_compatible():
    import inspect
    p = inspect.signature(fit_dixon_coles).parameters
    assert p["half_life_days"].default == 730
    assert p["max_iter"].default == 2000
    # optional params must default to off, so an unconfigured fit is the plain one
    assert p["competition_weights"].default is None
    assert p["default_competition_weight"].default == 0.2
    assert p["prior_net"].default is None
    assert p["prior_strength"].default == 0.0


def _toy_matches(n_matches=60, n_teams=6, seed=3):
    pytest = __import__("pytest")
    np = pytest.importorskip("numpy")
    pd = __import__("pandas")
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(n_teams)]
    rows = []
    for k in range(n_matches):
        h, a = rng.choice(n_teams, size=2, replace=False)
        rows.append({"date": f"2024-{1 + k % 12:02d}-{1 + k % 28:02d}", "competition": "Friendly",
                     "home_team": teams[h], "away_team": teams[a],
                     "home_score": int(rng.poisson(1.4)), "away_score": int(rng.poisson(1.1))})
    return pd.DataFrame(rows)


def test_dc_objective_gradient_matches_finite_differences():
    pytest = __import__("pytest")
    pytest.importorskip("scipy")
    import numpy as np
    from scipy.optimize import approx_fprime
    from src.models.dixon_coles import _dc_objective

    rng = np.random.default_rng(0)
    n, m = 5, 40
    hi = rng.integers(0, n, m); ai = (hi + 1 + rng.integers(0, n - 1, m)) % n
    hs = rng.poisson(1.3, m); as_ = rng.poisson(1.0, m)
    w = rng.uniform(0.2, 1.0, m)
    for prior in (None, rng.normal(0, 0.4, n)):
        fun = _dc_objective(hi, ai, hs, as_, w, n, prior, 1.5 if prior is not None else 0.0)
        p = rng.normal(0, 0.2, 2 * n + 2)
        p[-1] = -0.05                                     # rho in the sane range
        val, grad = fun(p)
        num = approx_fprime(p, lambda q: fun(q)[0], 1e-7)
        assert np.allclose(grad, num, atol=1e-4), f"max diff {np.abs(grad - num).max()}"


def test_fit_actually_converges_with_analytic_gradient():
    pytest = __import__("pytest")
    pytest.importorskip("scipy")
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        params = fit_dixon_coles(_toy_matches())
    assert not any("converge" in str(w.message).lower() for w in caught)
    assert set(params) == {"attack", "defence", "home_adv", "rho"}
