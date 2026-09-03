"""Choose the squad-strength coefficient by held-out RPS on recent matches.

Restricted to matches since 2023, where the 2026 rosters are a reasonable stand-in
for the squads that actually played. A coefficient of 0 leaves the goal model
untouched. Chosen on match-level accuracy, never on title odds.

Two limits worth stating plainly: the squads are proxied rather than historical,
and the signal is strongest for World Cup teams, which are not the whole test set.
Writes reports/goal_backtest_squad.md."""
from pathlib import Path
import pandas as pd
from src.config import load_params
from src.evaluation.goal_backtest import squad_coef_walkforward

COEF_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
N_SPLITS = 3
SINCE = "2023-01-01"


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    ps = pd.read_parquet("data/processed/player_stats.parquet")
    sq = pd.read_csv("data/processed/squad_coverage.csv")
    cfg = load_params()
    dc = cfg["models"]["dixon_coles"]

    res = squad_coef_walkforward(
        m, r, sq, ps, n_splits=N_SPLITS, coef_grid=COEF_GRID,
        competition_weights=dc["competition_weights"],
        default_competition_weight=dc.get("default_competition_weight", 0.2),
        prior_strength=dc["prior_strength"], beta=dc.get("prior_scale", 0.83),
        since=SINCE, months=cfg.get("recency_months", 24))

    agg = res.groupby("coef")[["rps", "log_loss", "brier"]].mean().sort_values("rps")
    ntest = res.groupby("split")["n_test"].first().to_dict()
    skipped = res.groupby("split")["n_skipped"].first().to_dict()
    best = float(agg.reset_index().iloc[0]["coef"])
    incumbent = float(agg.loc[0.0, "rps"])
    win = best != 0.0 and agg.loc[best, "rps"] < incumbent

    lines = [
        "# Squad-coefficient calibration (held-out RPS, recent window)",
        "",
        f"Window: matches since {SINCE} | n_splits={N_SPLITS} | coef_grid={COEF_GRID}",
        f"Per-split test sizes: {ntest} | skipped (unseen team): {skipped}",
        "Variant: deployed `both` (competition weighting + Elo prior lam=1.5) + squad bump.",
        "Limitations: 2026-roster proxy for historical squads; signal affects WC teams only.",
        "",
        "Mean held-out metrics per coef, sorted by RPS (lower = better):",
        "", "```", agg.to_string(), "```", "",
        f"## Selection: coef = {best if win else 0.0} "
        f"({'beats' if win else 'does NOT beat'} coef=0 incumbent {incumbent:.6f}). "
        "Chosen on held-out RPS, never on title odds.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/goal_backtest_squad.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
