"""Choose the Elo-anchor weight by held-out ranked probability score.

Scored only on matches between teams from different confederations, since 2014.
The gap between goal-scoring strength and Elo is only visible when teams from
different regions meet; inside a confederation the two measures largely agree.
A weight of 0 leaves the Dixon-Coles fit untouched.

The weight is chosen on match-level accuracy, never on how the title odds look.
Decision rule: if every split holds at least ~150 cross-confederation matches,
take the weight with the lowest mean RPS. Below that the sample is too thin to
trust, so fall back to 0.7. Writes reports/elo_anchor_backtest.md."""
from pathlib import Path
import pandas as pd
from src.config import load_params
from src.evaluation.goal_backtest import elo_anchor_walkforward

WEIGHT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
N_SPLITS = 4
SINCE = "2014-01-01"


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    confeds_df = pd.read_csv("data/reference/confederations.csv")
    confeds = {row["team"]: row["confederation"]
               for _, row in confeds_df.iterrows()
               if pd.notna(row["confederation"]) and row["confederation"] != ""}
    cfg = load_params()
    dc = cfg["models"]["dixon_coles"]

    res = elo_anchor_walkforward(
        m, r, confeds, n_splits=N_SPLITS, weight_grid=WEIGHT_GRID,
        competition_weights=dc["competition_weights"],
        default_competition_weight=dc.get("default_competition_weight", 0.2),
        prior_strength=dc["prior_strength"], beta=dc.get("prior_scale", 0.83),
        since=SINCE)

    agg = res.groupby("weight")[["rps", "log_loss", "brier"]].mean().sort_values("rps")
    ncross = res.groupby("split")["n_cross"].first().to_dict()
    skipped = res.groupby("split")["n_skipped"].first().to_dict()
    best = float(agg.reset_index().iloc[0]["weight"])
    incumbent = float(agg.loc[0.0, "rps"])
    win = best != 0.0 and agg.loc[best, "rps"] < incumbent

    # Decision gate: per-split n_cross must be sufficient
    min_n_cross = int(res.groupby("split")["n_cross"].first().min())
    sufficient = all(n >= 150 for n in ncross.values())

    if sufficient:
        selected = best if win else 0.0
        gate_note = f"RPS-selected weight = {selected} (cross-confed sample sufficient, n_cross ≥ 150 per split)"
    else:
        selected = 0.7
        gate_note = (f"Cross-confed sample too thin (min n_cross={min_n_cross} < 150). "
                     "Falling back to face-validity default weight=0.7. "
                     "This is NOT odds-tuning; the gate is documented above.")

    lines = [
        "# Elo-anchor calibration (cross-confed held-out RPS)",
        "",
        f"Window: matches since {SINCE} | n_splits={N_SPLITS} | weight_grid={WEIGHT_GRID}",
        f"Per-split cross-confed test sizes: {ncross} | skipped (unseen/unmapped): {skipped}",
        f"Min n_cross per split: {min_n_cross} | Sufficient for RPS selection: {sufficient}",
        "Variant: deployed `both` (competition weighting + Elo prior λ=1.5) + Elo anchor at each weight.",
        "Scored ONLY on cross-confederation test matches (where goal-model-vs-Elo bias is visible).",
        "",
        "Mean held-out metrics per weight, sorted by RPS (lower = better):",
        "", "```", agg.to_string(), "```", "",
        f"## Selection: weight = {selected}",
        f"Incumbent (weight=0) RPS: {incumbent:.6f}",
        f"Best RPS weight: {best} ({'beats' if win else 'does NOT beat'} incumbent)",
        f"Gate: {gate_note}",
        "",
        "Chosen on RPS over cross-confederation matches, never on title odds.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/elo_anchor_backtest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
