# src/models/goals.py
import math
import numpy as np

def poisson_pmf(k, lam):
    """Poisson(lam) probability of exactly k goals. Pure (stdlib factorial)."""
    return math.exp(-lam) * lam ** k / math.factorial(k)

def dc_tau(h, a, lam_h, lam_a, rho):
    """Dixon-Coles low-score dependence factor for scoreline (h, a). Inflates 1-1
    /0-0 dependence and the 1-0/0-1 cells; identity (1.0) outside the 0-1 corner."""
    if h == 0 and a == 0:
        return 1.0 - lam_h * lam_a * rho
    if h == 0 and a == 1:
        return 1.0 + lam_h * rho
    if h == 1 and a == 0:
        return 1.0 + lam_a * rho
    if h == 1 and a == 1:
        return 1.0 - rho
    return 1.0

def _poisson_pmf_vec(lam, max_goals):
    """Vectorized Poisson pmf over k = 0..max_goals via the log form
    exp(k*ln(lam) - lam - ln k!), equal to poisson_pmf cell-by-cell but built in
    one numpy expression (the Monte-Carlo hot path builds ~1M of these)."""
    if lam <= 0.0:
        out = np.zeros(max_goals + 1)
        out[0] = 1.0
        return out
    ks = np.arange(max_goals + 1)
    log_fact = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, max_goals + 1))))) \
        if max_goals > 0 else np.zeros(1)
    return np.exp(ks * math.log(lam) - lam - log_fact)


def score_matrix(lam_h, lam_a, rho=0.0, max_goals=10):
    """(max_goals+1) x (max_goals+1) scoreline matrix M[h, a]: independent
    Poisson(lam_h) x Poisson(lam_a) with the Dixon-Coles correction applied to the
    four low-score cells, renormalised to sum to 1. Pure numpy."""
    M = np.outer(_poisson_pmf_vec(lam_h, max_goals), _poisson_pmf_vec(lam_a, max_goals))
    for h in (0, 1):
        for a in (0, 1):
            M[h, a] *= dc_tau(h, a, lam_h, lam_a, rho)
    return M / M.sum()

def matrix_to_wdl(matrix):
    """Scoreline matrix -> (p_home, p_draw, p_away). Rows = home goals, cols = away
    goals, so home win is the strict lower triangle, draw the diagonal."""
    p_home = float(np.tril(matrix, -1).sum())
    p_draw = float(np.trace(matrix))
    p_away = float(np.triu(matrix, 1).sum())
    return p_home, p_draw, p_away

def expected_goals_from_matrix(matrix):
    """Expected goals (home, away) implied by the (truncated) scoreline matrix."""
    goals = np.arange(matrix.shape[0])
    return float((matrix.sum(axis=1) * goals).sum()), float((matrix.sum(axis=0) * goals).sum())
