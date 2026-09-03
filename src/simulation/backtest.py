def backtest_tournament(matches, fmt, groups, tournament_start, tournament="2022",
                        n_runs=2000, seed=42):
    """THIN acceptance-gate harness (optional scipy): fit Dixon-Coles on matches
    STRICTLY BEFORE `tournament_start` (leakage-safe, never sees a tournament
    result), then Monte-Carlo the historical bracket through the same engine the
    2026 forward sim uses. `groups` is {group: [team, ...]} for the historical
    tournament; `fmt` is its config/tournaments.yaml entry. Returns the reach-
    probability frame to compare against actual outcomes (manual smoke below).
    Not unit-tested (scipy fit); import-smoke only."""
    import numpy as np
    import pandas as pd
    from src.models.dixon_coles import fit_dixon_coles
    from src.simulation.montecarlo import run_simulation

    pre = matches[pd.to_datetime(matches["date"]) < pd.Timestamp(tournament_start)]
    params = fit_dixon_coles(pre)
    return run_simulation(
        fmt, groups, params, fmt["seeding"], n_runs=n_runs,
        rng=np.random.default_rng(seed), hosts=fmt.get("hosts", ()),
        jitter=0.0, rho=params.get("rho", 0.0), tournament=tournament,
    )
