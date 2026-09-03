import numpy as np
from src.simulation.params import sample_params, apply_squad_bump

FITTED = {"attack": {"A": 0.2, "B": -0.1}, "defence": {"A": 0.1, "B": 0.0},
          "home_adv": 0.3, "rho": -0.05}


def test_zero_jitter_is_identity():
    out = sample_params(FITTED, np.random.default_rng(0), jitter=0.0)
    assert out["attack"] == FITTED["attack"] and out["defence"] == FITTED["defence"]
    assert out["home_adv"] == 0.3 and out["rho"] == -0.05


def test_jitter_perturbs_and_is_reproducible():
    a = sample_params(FITTED, np.random.default_rng(7), jitter=0.1)
    b = sample_params(FITTED, np.random.default_rng(7), jitter=0.1)
    assert a["attack"] == b["attack"]                       # same seed -> same draw
    assert a["attack"]["A"] != FITTED["attack"]["A"]        # actually perturbed
    assert a["home_adv"] == FITTED["home_adv"]              # passthrough


_PARAMS = {"attack": {"A": 0.2, "B": -0.1, "C": 0.5},
           "defence": {"A": 0.1, "B": 0.0, "C": 0.3}, "home_adv": 0.3, "rho": 0.0}


def test_coef_zero_is_identity():
    out = apply_squad_bump(_PARAMS, {"A": (1.0, 1.0), "B": (-1.0, -1.0)}, 0.0)
    assert out["attack"] == _PARAMS["attack"]
    assert out["defence"] == _PARAMS["defence"]


def test_positive_coef_raises_stronger_team():
    out = apply_squad_bump(_PARAMS, {"A": (1.0, 1.0), "B": (-1.0, -1.0)}, 0.5)
    assert out["attack"]["A"] > out["attack"]["B"]
    assert out["defence"]["A"] > out["defence"]["B"]


def test_team_without_strength_is_unchanged():
    out = apply_squad_bump(_PARAMS, {"A": (1.0, 1.0), "B": (-1.0, -1.0)}, 0.5)
    assert out["attack"]["C"] == 0.5 and out["defence"]["C"] == 0.3


def test_recentring_preserves_mean_of_bumped():
    out = apply_squad_bump(_PARAMS, {"A": (2.0, 1.0), "B": (1.0, 0.5)}, 1.0)
    orig_mean_atk = (_PARAMS["attack"]["A"] + _PARAMS["attack"]["B"]) / 2
    new_mean_atk = (out["attack"]["A"] + out["attack"]["B"]) / 2
    assert abs(new_mean_atk - orig_mean_atk) < 1e-9
    orig_mean_dfc = (_PARAMS["defence"]["A"] + _PARAMS["defence"]["B"]) / 2
    new_mean_dfc = (out["defence"]["A"] + out["defence"]["B"]) / 2
    assert abs(new_mean_dfc - orig_mean_dfc) < 1e-9


# --- apply_elo_anchor tests ---

from src.simulation.params import apply_elo_anchor

# net strengths: A=2.0, B=1.0, C=0.0 -> mean 1.0 -> centered A=+1, B=0, C=-1
_PE = {"attack": {"A": 1.0, "B": 0.5, "C": 0.0},
       "defence": {"A": 1.0, "B": 0.5, "C": 0.0}, "home_adv": 0.3, "rho": 0.0}


def test_elo_anchor_zero_is_identity():
    out = apply_elo_anchor(_PE, {"A": 2.0, "B": 0.0, "C": -2.0}, 0.0)
    assert out["attack"] == _PE["attack"] and out["defence"] == _PE["defence"]


def test_elo_anchor_weight_one_sets_centered_strength_to_target():
    tgt = {"A": 2.0, "B": 0.0, "C": -2.0}
    out = apply_elo_anchor(_PE, tgt, 1.0)
    mean = 1.0  # mean of base net strengths over A,B,C
    for t in ("A", "B", "C"):
        net = out["attack"][t] + out["defence"][t]
        assert abs((net - mean) - tgt[t]) < 1e-9


def test_elo_anchor_absent_team_unchanged():
    out = apply_elo_anchor(_PE, {"A": 2.0}, 1.0)   # only A targeted
    assert out["attack"]["B"] == 0.5 and out["defence"]["C"] == 0.0
