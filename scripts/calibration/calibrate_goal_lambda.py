"""Choose the goal model's shrinkage strength (lambda) by walk-forward backtest.

Match-level walk-forward over the recent window (since 2014), with beta fixed at
0.83, the empirical standard deviation of team strength. Four splits, each
training only on matches that precede the ones it scores. The winner is the
lambda with the lowest mean held-out RPS, log-loss breaking ties, and it has to
beat the unweighted baseline. Never chosen on title odds.

This one is slow: each fit takes several minutes on 12K matches, so the full grid
runs a couple of hours. Cut n_splits or use a later cutoff to iterate faster.
Writes reports/goal_backtest.md."""
from pathlib import Path
import pandas as pd

from src.config import load_params
from src.evaluation.goal_backtest import goalmodel_walkforward

BETA = 0.83                 # empirical sd(strength = atk+dfc) from the Task-4 smoke
N_SPLITS = 4
LAM_GRID = (0.5, 1.5, 4.0)


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    tw = load_params()["models"]["dixon_coles"]["competition_weights"]

    res = goalmodel_walkforward(m, r, n_splits=N_SPLITS, lam_grid=LAM_GRID,
                                competition_weights=tw, beta=BETA, since="2014-01-01")
    agg = (res.groupby(["variant", "lam"])[["rps", "log_loss", "brier"]]
              .mean().sort_values("rps"))
    skipped = res.groupby("split")["n_skipped"].first().to_dict()
    ntest = res.groupby("split")["n_test"].first().to_dict()

    lines = [
        "# Goal-model walk-forward backtest: choosing lambda",
        "",
        f"Window: matches since 2014 | n_splits={N_SPLITS} | beta(prior_scale)={BETA} | lam_grid={LAM_GRID}",
        f"Per-split test sizes: {ntest} | skipped (unseen team): {skipped}",
        "",
        "Mean held-out metrics per (variant, lambda), sorted by RPS (lower = better):",
        "",
        "```",
        agg.to_string(),
        "```",
        "",
        "Selection: pick the lambda with the lowest mean RPS; confirm",
        "weight/both <= baseline. If nothing beats baseline, lock prior_strength=0.0 and",
        "keep weighting only. Chosen on RPS, never on title odds.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/goal_backtest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
