import math
from src.models.goals import score_matrix, matrix_to_wdl
from src.simulation.shootout import shootout_winner


def match_lambdas(params, home, away, hosts=()):
    """(lam_home, lam_away) for a fixture from fitted Dixon-Coles params, applying
    home_adv ONLY when the nominal home team is a tournament host playing
    in-country; neutral otherwise. Works for any team pair (knockout-ready).
    Unseen teams (not in params) regress to league-average (atk=0, def=0)."""
    atk, dfc = params["attack"], params["defence"]
    ha = params["home_adv"] if home in set(hosts) else 0.0
    atk_h = atk.get(home, 0.0)
    dfc_a = dfc.get(away, 0.0)
    atk_a = atk.get(away, 0.0)
    dfc_h = dfc.get(home, 0.0)
    return (math.exp(atk_h - dfc_a + ha), math.exp(atk_a - dfc_h))


def match_probs(lam_h, lam_a, rho=0.0, max_goals=10):
    """(p_home, p_draw, p_away) from the Dixon-Coles goal model. The single
    swappable W/D/L seam: a later CatBoost-W/D/L blend can replace this body
    without touching the tournament loop."""
    return matrix_to_wdl(score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals))


def knockout_winner(lam_h, lam_a, rng, rho=0.0, max_goals=10, et_factor=1.0 / 3.0,
                    p_home_shootout=0.5):
    """Resolve a knockout regulation draw: sample a 30-minute extra time from the
    same goal model at `et_factor` * lam (ET is a third of regulation, so the
    stronger side keeps a proportional scoring edge), then fall back to the
    near-coin-flip shootout only if extra time is also level. This replaces the
    pure coin flip that gave favorites zero edge in any drawn knockout."""
    M = score_matrix(lam_h * et_factor, lam_a * et_factor, rho=rho, max_goals=max_goals)
    k = rng.choice(M.size, p=M.ravel())
    hg, ag = divmod(k, M.shape[1])
    if hg > ag:
        return "home"
    if hg < ag:
        return "away"
    return shootout_winner(rng, p_home=p_home_shootout)


def simulate_match(lam_h, lam_a, rng, rho=0.0, max_goals=10, knockout=False, p_home_shootout=0.5):
    """Sample one match from the Dixon-Coles scoreline matrix. Returns
    (home_goals, away_goals, winner): winner is 'home'/'away', or 'draw' for a
    group-stage tie. A knockout regulation draw is resolved by extra time + the
    shootout model via `knockout_winner` (winner becomes 'home'/'away'; the
    sampled regulation goals are still returned)."""
    M = score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)
    k = rng.choice(M.size, p=M.ravel())
    hg, ag = (int(x) for x in divmod(k, M.shape[1]))
    if hg > ag:
        winner = "home"
    elif hg < ag:
        winner = "away"
    elif knockout:
        winner = knockout_winner(lam_h, lam_a, rng, rho=rho, max_goals=max_goals,
                                 p_home_shootout=p_home_shootout)
    else:
        winner = "draw"
    return hg, ag, winner
