# src/models/dixon_coles.py
import math
import numpy as np
import pandas as pd


def _dc_objective(hi, ai, hs, as_, w, n, prior_vec, prior_strength):
    """Build fun(p) -> (nll, grad) for the weighted Dixon-Coles likelihood with the
    optional Elo net-strength prior. Fully vectorized (mask-based tau) with the
    ANALYTIC gradient, the numeric-gradient fit silently exhausted L-BFGS-B's
    maxfun budget after ~17 iterations on the full dataset (731 evals/gradient).

    p = [a_raw(n), dfc(n), home_adv, rho]; atk = a_raw - mean(a_raw)."""
    from scipy.special import gammaln

    hi = np.asarray(hi); ai = np.asarray(ai)
    hs = np.asarray(hs, dtype=float); as_ = np.asarray(as_, dtype=float)
    w = np.asarray(w, dtype=float)
    lf = gammaln(hs + 1) + gammaln(as_ + 1)
    m00 = (hs == 0) & (as_ == 0)
    m01 = (hs == 0) & (as_ == 1)
    m10 = (hs == 1) & (as_ == 0)
    m11 = (hs == 1) & (as_ == 1)
    pv = None if prior_vec is None else np.asarray(prior_vec, dtype=float)

    def fun(p):
        a_raw = p[:n]
        atk = a_raw - a_raw.mean()
        dfc = p[n:2 * n]
        ha, rho = p[2 * n], p[2 * n + 1]
        eta_h = atk[hi] - dfc[ai] + ha                    # ln(lam): keep the likelihood in
        eta_a = atk[ai] - dfc[hi]                         # log space so lam underflow (exp of
        lam_h = np.exp(eta_h)                             # a very negative eta during line
        lam_a = np.exp(eta_a)                             # search) never hits log(0)

        tau_raw = np.ones_like(lam_h)
        tau_raw[m00] = 1.0 - lam_h[m00] * lam_a[m00] * rho
        tau_raw[m01] = 1.0 + lam_h[m01] * rho
        tau_raw[m10] = 1.0 + lam_a[m10] * rho
        tau_raw[m11] = 1.0 - rho
        tau = np.clip(tau_raw, 1e-12, None)
        active = tau_raw > 1e-12                          # where the clip binds the value is
                                                          # constant -> zero (sub)gradient there

        ll = (hs * eta_h - lam_h) + (as_ * eta_a - lam_a) - lf + np.log(tau)
        nll = -float(np.sum(w * ll))

        # d tau / d lam and d tau / d rho on the four low-score cells (masked by `active`
        # so value and gradient stay consistent, an inconsistent pair breaks line search)
        dt_dlh = np.zeros_like(lam_h); dt_dla = np.zeros_like(lam_h); dt_dr = np.zeros_like(lam_h)
        dt_dlh[m00] = -lam_a[m00] * rho
        dt_dla[m00] = -lam_h[m00] * rho
        dt_dr[m00] = -lam_h[m00] * lam_a[m00]
        dt_dlh[m01] = rho
        dt_dr[m01] = lam_h[m01]
        dt_dla[m10] = rho
        dt_dr[m10] = lam_a[m10]
        dt_dr[m11] = -1.0
        dt_dlh *= active; dt_dla *= active; dt_dr *= active

        gh = w * (hs - lam_h + lam_h * dt_dlh / tau)      # d(sum w*ll)/d ln(lam_h), per match
        ga = w * (as_ - lam_a + lam_a * dt_dla / tau)
        g_atk = np.bincount(hi, gh, n) + np.bincount(ai, ga, n)
        g_dfc = -(np.bincount(ai, gh, n) + np.bincount(hi, ga, n))
        g_ha = float(gh.sum())
        g_rho = float(np.sum(w * dt_dr / tau))

        grad_atk = -g_atk
        grad_dfc = -g_dfc
        if pv is not None and prior_strength:
            s = atk + dfc
            r = (s - s.mean()) - pv
            gpen = 2.0 * float(prior_strength) * (r - r.mean())   # centering projector
            nll += float(prior_strength) * float(np.sum(r ** 2))
            grad_atk = grad_atk + gpen
            grad_dfc = grad_dfc + gpen

        grad = np.empty_like(p)
        grad[:n] = grad_atk - grad_atk.mean()             # chain through atk centering
        grad[n:2 * n] = grad_dfc
        grad[2 * n] = -g_ha
        grad[2 * n + 1] = -g_rho
        return nll, grad

    return fun

def team_lambdas(params, home, away):
    """Expected goals (lam_home, lam_away) from fitted Dixon-Coles params.
    params = {'attack': {team: a}, 'defence': {team: d}, 'home_adv': float}.
    Pure (no fitting), so unit-testable with hand-built params."""
    atk, dfc, ha = params["attack"], params["defence"], params["home_adv"]
    lam_h = math.exp(atk[home] - dfc[away] + ha)
    lam_a = math.exp(atk[away] - dfc[home])
    return lam_h, lam_a

def fit_dixon_coles(matches, half_life_days=730, max_iter=2000,
                    competition_weights=None, default_competition_weight=0.2,
                    prior_net=None, prior_strength=0.0):
    """THIN wrapper (optional scipy): Dixon-Coles (1997) MLE fit of per-team
    attack/defence + home advantage + low-score rho over historical scorelines,
    with exponential time-decay weighting (recency, NOT leakage - never uses a
    match's own future score). scipy is an optional runtime dep, lazy-imported;
    not fixture-tested. Returns the params dict consumed by `team_lambdas` and
    `goals.score_matrix`.

    Optional refinements (all default to today's exact behavior):
      - `competition_weights`: an ordered [substring, weight] tier list; when given
        (and `matches` has a `competition` column) the per-match recency weight is
        multiplied by the competition-tier weight, so friendlies/minor cups count
        less (see `competition.competition_weights`).
      - `prior_net` + `prior_strength` (lambda): an Elo-anchored shrinkage prior that
        pulls each team's overall strength (atk + dfc -- the quantity that drives match
        supremacy, since a higher defence rating lowers the opponent's goals) toward
        `prior_net` via the penalty lambda * sum((centered_strength - prior_net)**2).
        The strength is mean-centred so the penalty constrains only relative strengths,
        never the overall goal level. `prior_net` may be a {team: value} dict or a
        length-n array in sorted(teams) order. lambda=0 (default) -> no penalty,
        byte-identical to the unpenalised fit."""
    from scipy.optimize import minimize

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    teams = sorted(set(df["home_team"]) | set(df["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    hi = df["home_team"].map(idx).to_numpy()
    ai = df["away_team"].map(idx).to_numpy()
    hs = df["home_score"].to_numpy(dtype=int)
    as_ = df["away_score"].to_numpy(dtype=int)
    if half_life_days:
        age = (df["date"].max() - df["date"]).dt.days.to_numpy()
        w = 0.5 ** (age / float(half_life_days))
    else:
        w = np.ones(len(df))
    if competition_weights is not None and "competition" in df.columns:
        from src.models.competition import competition_weights as _comp_w
        w = w * _comp_w(df["competition"], competition_weights, default_competition_weight)

    if prior_strength and prior_net is not None:     # Elo-anchored net-strength prior
        if isinstance(prior_net, dict):
            prior_vec = np.array([float(prior_net.get(t, 0.0)) for t in teams])
        else:
            prior_vec = np.asarray(prior_net, dtype=float)
            assert prior_vec.shape == (n,), "prior_net must align to sorted(teams)"
    else:
        prior_vec = None

    neg_loglik = _dc_objective(hi, ai, hs, as_, w, n, prior_vec, prior_strength)
    x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.25, -0.05]])
    res = minimize(neg_loglik, x0, method="L-BFGS-B", jac=True,
                   options={"maxiter": max_iter})
    if not res.success:
        import warnings
        warnings.warn(
            f"fit_dixon_coles did NOT converge ({res.message!r}, nit={res.nit}); "
            "returned params are the optimizer's last iterate - treat with suspicion",
            RuntimeWarning, stacklevel=2,
        )
    atk = res.x[:n] - res.x[:n].mean()
    dfc = res.x[n:2 * n]
    return {
        "attack": {t: float(atk[idx[t]]) for t in teams},
        "defence": {t: float(dfc[idx[t]]) for t in teams},
        "home_adv": float(res.x[2 * n]),
        "rho": float(res.x[2 * n + 1]),
    }
