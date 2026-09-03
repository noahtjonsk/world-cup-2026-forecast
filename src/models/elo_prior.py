import numpy as np

def elo_prior_net(elo_by_team, teams, beta=0.35):
    """Elo-anchored prior target for DC team strength (the centered atk + dfc that
    drives match supremacy), aligned to `teams`.
    Standardize the Elo of the PRESENT teams to mean 0 / unit sd (ddof=0), scale by
    `beta`, and place each team's value at its index in `teams`; teams absent from
    `elo_by_team` -> 0.0 (neutral = league average, matching the unseen-team handling
    elsewhere). Zero/undefined variance (<2 present teams or constant Elo) -> all
    zeros (no divide-by-zero). Pure numpy. Mean-centred like the fit's `atk`."""
    present = [t for t in teams if t in elo_by_team]
    out = np.zeros(len(teams), dtype=float)
    if len(present) < 2:
        return out
    vals = np.array([float(elo_by_team[t]) for t in present])
    sd = vals.std()                                    # ddof=0
    if sd <= 0:
        return out
    z = (vals - vals.mean()) / sd
    pos = {t: i for i, t in enumerate(teams)}
    for t, zi in zip(present, z):
        out[pos[t]] = beta * float(zi)
    return out

def elo_prior_net_asof(ratings, teams, cutoff, beta=0.35):
    """Leakage-safe convenience: build the {team: elo} seed strictly before `cutoff`
    via seed_from_ratings, then delegate to elo_prior_net. Use in each backtest split
    with cutoff = test-window start, and in production with the WC cutoff. The import
    is lazy so this module carries no new top-level deps."""
    from src.states.elo_update import seed_from_ratings
    return elo_prior_net(seed_from_ratings(ratings, cutoff), teams, beta=beta)
