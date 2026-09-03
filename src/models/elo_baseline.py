# src/models/elo_baseline.py
import numpy as np

def elo_expected_score(elo_diff, home_adv=65.0, neutral=False):
    """Elo win-expectancy of the home team (incl. half-credit for draws):
    logistic on (elo_diff + home_adv), home_adv zeroed at a neutral venue."""
    d = elo_diff + (0.0 if neutral else home_adv)
    return 1.0 / (1.0 + 10 ** (-d / 400.0))

def elo_wdl_probs(elo_diff, home_adv=65.0, neutral=False, draw_base=0.30):
    """Closed-form W/D/L from Elo. Draw mass = draw_base * 4*We*(1-We) (peaks at
    even matches, ->0 at extremes); split symmetrically so We = H + D/2 and the
    three probabilities sum to 1. Non-negative for draw_base <= 0.5."""
    we = elo_expected_score(elo_diff, home_adv, neutral)
    p_draw = draw_base * 4.0 * we * (1.0 - we)
    return {"H": we - p_draw / 2.0, "D": p_draw, "A": (1.0 - we) - p_draw / 2.0}

def predict_proba(elo_diff, neutral, home_adv=65.0, draw_base=0.30):
    """Vectorized baseline: arrays `elo_diff`, `neutral` (bool) -> (n,3) array in
    fixed [H, D, A] order (matches src.evaluation.metrics.CLASSES)."""
    elo_diff = np.asarray(elo_diff, dtype=float)
    neutral = np.asarray(neutral, dtype=bool)
    d = elo_diff + np.where(neutral, 0.0, home_adv)
    we = 1.0 / (1.0 + 10 ** (-d / 400.0))
    p_draw = draw_base * 4.0 * we * (1.0 - we)
    return np.column_stack([we - p_draw / 2.0, p_draw, (1.0 - we) - p_draw / 2.0])

def fit_draw_base(elo_diff, neutral, y_true, home_adv=65.0, grid=None):
    """Tune the one free baseline parameter (the draw band) on a TRAIN fold by
    grid-minimising log-loss. Pure numpy, the Elo baseline's only 'training'."""
    from src.evaluation.metrics import log_loss
    if grid is None:
        grid = np.linspace(0.05, 0.45, 9)
    best, best_ll = float(grid[0]), float("inf")
    for db in grid:
        ll = log_loss(y_true, predict_proba(elo_diff, neutral, home_adv=home_adv, draw_base=float(db)))
        if ll < best_ll:
            best_ll, best = ll, float(db)
    return best
