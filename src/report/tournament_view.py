import pandas as pd
from src.simulation.montecarlo import champion_odds

DEFAULT_ROUND_ORDER = ["R32", "R16", "QF", "SF", "F", "W"]


def progression_table(results, tournament="2026", round_order=DEFAULT_ROUND_ORDER):
    """Team by round reach-probability table for one tournament, best team first.

    `results` is the simulation_results frame. Returns one row per team, with a column
    for each round present in `round_order`, ordered by title probability."""
    df = results[results["tournament"] == tournament]
    cols = [r for r in round_order if r in set(df["round"])]
    wide = df.pivot_table(index="team", columns="round", values="prob", aggfunc="first")
    order = list(champion_odds(df)["team"])                 # teams by P(win) desc
    wide = wide.reindex(index=order, columns=cols)
    return wide.reset_index()[["team"] + cols]


def qualification_view(results, tournament="2026", teams=None, round_order=DEFAULT_ROUND_ORDER):
    """Tidy [team, round, prob, ci_low, ci_high] for one tournament, with `round`
    as an ordered categorical (so charts order R32->...->W) and sorted by team then
    round. Optionally restrict to `teams`. A leakage-free view over simulation_results."""
    df = results[results["tournament"] == tournament].copy()
    if teams is not None:
        df = df[df["team"].isin(teams)]
    present = [r for r in round_order if r in set(df["round"])]
    df["round"] = pd.Categorical(df["round"], categories=present, ordered=True)
    df = df.sort_values(["team", "round"]).reset_index(drop=True)
    return df[["team", "round", "prob", "ci_low", "ci_high"]]
