# tests/models/test_goals.py
import math
from src.models.goals import (
    poisson_pmf, dc_tau, score_matrix, matrix_to_wdl, expected_goals_from_matrix,
)

def test_poisson_pmf_values():
    assert abs(poisson_pmf(0, 1.0) - math.exp(-1.0)) < 1e-12
    assert abs(poisson_pmf(2, 2.0) - 2.0 * math.exp(-2.0)) < 1e-12        # e^-2 * 2^2/2!

def test_dc_tau_low_score_corrections():
    assert dc_tau(0, 0, 1.0, 1.0, 0.1) == 1.0 - 1.0 * 1.0 * 0.1
    assert dc_tau(0, 1, 1.0, 1.0, 0.1) == 1.0 + 1.0 * 0.1
    assert dc_tau(1, 0, 1.0, 1.0, 0.1) == 1.0 + 1.0 * 0.1
    assert dc_tau(1, 1, 1.0, 1.0, 0.1) == 1.0 - 0.1
    assert dc_tau(2, 2, 1.0, 1.0, 0.1) == 1.0                             # untouched outside low scores

def test_score_matrix_normalised_and_symmetric():
    M = score_matrix(1.0, 1.0, rho=0.0, max_goals=10)
    assert abs(M.sum() - 1.0) < 1e-9
    ph, pdr, pa = matrix_to_wdl(M)
    assert abs(ph + pdr + pa - 1.0) < 1e-9
    assert abs(ph - pa) < 1e-9                                           # equal lambdas -> symmetric

def test_expected_goals_recovered():
    M = score_matrix(1.5, 0.8, rho=0.0, max_goals=15)
    egh, ega = expected_goals_from_matrix(M)
    assert abs(egh - 1.5) < 1e-3 and abs(ega - 0.8) < 1e-3


def test_score_matrix_equals_explicit_pmf_construction():
    # characterization: the (vectorizable) matrix must equal the literal
    # outer-product of poisson_pmf values with dc_tau corrections, renormalised.
    import numpy as np
    from src.models.goals import score_matrix, poisson_pmf, dc_tau
    for lam_h, lam_a, rho in [(1.4, 1.1, 0.0), (2.7, 0.4, -0.08), (0.6, 0.6, 0.1)]:
        h = np.array([poisson_pmf(i, lam_h) for i in range(11)])
        a = np.array([poisson_pmf(j, lam_a) for j in range(11)])
        M = np.outer(h, a)
        for i in (0, 1):
            for j in (0, 1):
                M[i, j] *= dc_tau(i, j, lam_h, lam_a, rho)
        M = M / M.sum()
        assert np.allclose(score_matrix(lam_h, lam_a, rho=rho, max_goals=10), M, atol=1e-12)
