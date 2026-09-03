import numpy as np
import pandas as pd
from src.schema import CANON_SIM_RESULT_COLS, CANON_BRACKET_COLS
from src.simulation.params import sample_params
from src.simulation.tournament import simulate_tournament


def _stage_labels(rounds):
    """Stage index -> label: 0=group, 1..len=knockout rounds, len+1='W' (champion)."""
    labels = {0: "group"}
    for i, r in enumerate(rounds):
        labels[i + 1] = r
    labels[len(rounds) + 1] = "W"
    return labels


def run_simulation(fmt, groups, params, seeding, n_runs, rng, hosts=(),
                   jitter=0.0, rho=0.0, max_goals=10, n_boot=1000, tournament="2026",
                   collect_bracket=False):
    """Monte-Carlo the tournament n_runs times with per-run parameter resampling
    (uncertainty propagation). Returns a tidy frame (CANON_SIM_RESULT_COLS) of each
    team's probability of REACHING each knockout stage (and winning), with
    bootstrap-over-runs 95% credible intervals. Pure numpy/pandas (seeded rng).

    collect_bracket=True additionally aggregates knockout slot occupancy, counts
    per (round, match_idx, home, away) pairing + home wins (CANON_BRACKET_COLS) 
    and returns (results, bracket). Recording is pure (consumes no rng), so the
    results frame is identical either way."""
    rounds = fmt["rounds"]
    labels = _stage_labels(rounds)
    teams = [t for tm in groups.values() for t in tm]
    tidx = {t: i for i, t in enumerate(teams)}
    n_stages = len(rounds) + 1                                # report stages 1..n_stages

    pair_counts = {}                                          # (round, idx, home, away) -> [n, home_wins]
    reach = np.zeros((n_runs, len(teams)), dtype=int)
    for r in range(n_runs):
        p = sample_params(params, rng, jitter=jitter)
        log = [] if collect_bracket else None
        _, reached = simulate_tournament(fmt, groups, p, seeding, rng,
                                         hosts=hosts, rho=rho, max_goals=max_goals,
                                         bracket_log=log)
        for t, s in reached.items():
            reach[r, tidx[t]] = s
        if collect_bracket:
            for rname, mi, home, away, winner in log:
                c = pair_counts.setdefault((rname, mi, home, away), [0, 0])
                c[0] += 1
                if winner == home:
                    c[1] += 1

    rows = []
    for s in range(1, n_stages + 1):
        hit = (reach >= s).astype(float)                      # (n_runs, n_teams) indicator
        prob = hit.mean(axis=0)
        boot = np.empty((n_boot, len(teams)))
        for b in range(n_boot):                               # memory-safe bootstrap over runs
            idx = rng.integers(0, n_runs, size=n_runs)
            boot[b] = hit[idx].mean(axis=0)
        lo = np.percentile(boot, 2.5, axis=0)
        hi = np.percentile(boot, 97.5, axis=0)
        for t in teams:
            i = tidx[t]
            rows.append({"tournament": tournament, "team": t, "round": labels[s],
                         "prob": float(prob[i]), "ci_low": float(lo[i]), "ci_high": float(hi[i])})
    results = pd.DataFrame(rows, columns=CANON_SIM_RESULT_COLS)
    if not collect_bracket:
        return results
    bracket = pd.DataFrame(
        [{"tournament": tournament, "round": rname, "match_idx": mi,
          "home_team": h, "away_team": a, "n": c[0], "home_wins": c[1]}
         for (rname, mi, h, a), c in pair_counts.items()],
        columns=CANON_BRACKET_COLS,
    ).sort_values(["round", "match_idx", "n"], ascending=[True, True, False]).reset_index(drop=True)
    return results, bracket


def champion_odds(results, top=None):
    """Title ('W') rows sorted by probability (desc), optionally the top-N. A pure
    view over a run_simulation result frame."""
    w = results[results["round"] == "W"].sort_values("prob", ascending=False).reset_index(drop=True)
    return w.head(top) if top else w
