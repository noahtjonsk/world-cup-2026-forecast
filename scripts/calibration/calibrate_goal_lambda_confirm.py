"""Narrow the lambda search around the optimum found by calibrate_goal_lambda.

The coarse grid (0.5, 1.5, 4.0) peaked at 1.5 and got worse by 4.0. This repeats
the same walk-forward over a denser local grid (1.0, 1.5, 2.0), scoring only the
deployed configuration of competition weighting plus the Elo prior. That pins the
optimum without re-running the full two-hour grid. Lambda 1.5 appears in both
runs, so it doubles as a reproducibility check against reports/goal_backtest.md.

Chosen on held-out RPS, never on title odds.
Writes reports/goal_backtest_confirm.md."""
from pathlib import Path
import pandas as pd

from src.config import load_params
from src.evaluation.goal_backtest import goalmodel_walkforward

BETA = 0.83                 # empirical sd(strength = atk+dfc) from the Task-4 smoke
N_SPLITS = 3                # match the existing reports/goal_backtest.md
LAM_GRID = (1.0, 1.5, 2.0)  # dense local grid around the 1.5 peak


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    tw = load_params()["models"]["dixon_coles"]["competition_weights"]

    res = goalmodel_walkforward(m, r, n_splits=N_SPLITS, lam_grid=LAM_GRID,
                                competition_weights=tw, beta=BETA, since="2014-01-01",
                                variants={"both"})
    agg = (res.groupby(["variant", "lam"])[["rps", "log_loss", "brier"]]
              .mean().sort_values("rps"))
    skipped = res.groupby("split")["n_skipped"].first().to_dict()
    ntest = res.groupby("split")["n_test"].first().to_dict()
    best_lam = float(agg.reset_index().iloc[0]["lam"])

    lines = [
        "# Lambda confirm grid (weighting plus prior)",
        "",
        f"Window: matches since 2014 | n_splits={N_SPLITS} | beta(prior_scale)={BETA} | lam_grid={LAM_GRID}",
        f"Per-split test sizes: {ntest} | skipped (unseen team): {skipped}",
        "Variant: `both` only (competition weighting + Elo prior), the deployed config.",
        "",
        "Mean held-out metrics per lambda, sorted by RPS (lower = better):",
        "",
        "```",
        agg.to_string(),
        "```",
        "",
        f"## Selection: lambda = {best_lam}, the lowest mean held-out RPS. Chosen on RPS, never on title odds.",
        "",
        "Cross-check: `both` lam=1.5 should reproduce ~0.174688 from reports/goal_backtest.md.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/goal_backtest_confirm.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
