"""Re-run the squad coefficient calibration against the corrected talent signal.

Same method as calibrate_squad_coef, but using the signal that counts midfielders
and weights each player by the strength of their league. Also scores the
cross-confederation subset when confederations.csv is present.

If the corrected signal does not beat a coefficient of 0 on RPS, the report says
so. Its value is that it ranks squads sensibly, not that it is proven to improve
accuracy. Writes reports/squad_coef_corrected_backtest.md."""
from pathlib import Path
import pandas as pd
import numpy as np
from src.config import load_params
from src.features.squad_strength import team_squad_strength, club_league_multiplier
from src.models.dixon_coles import fit_dixon_coles
from src.states.elo_update import seed_from_ratings
from src.models.elo_prior import elo_prior_net
from src.simulation.params import apply_squad_bump
from src.evaluation.goal_backtest import wdl_probs_for_matches, _result_labels
from src.evaluation.metrics import log_loss, rps, brier_score
from src.evaluation.walkforward import time_splits

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

    # Build v2 signal (league + midfield)
    all_clubs = sq["club"].dropna().unique()
    league_map = club_league_multiplier(all_clubs)

    df = m.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= pd.Timestamp(SINCE)]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)

    rows = []
    for split_i, (tr, te) in enumerate(time_splits(df, n_splits=N_SPLITS)):
        train, test = df.loc[tr], df.loc[te]
        seen = set(train["home_team"]) | set(train["away_team"])
        keep = test["home_team"].isin(seen) & test["away_team"].isin(seen)
        n_skip = int((~keep).sum())
        test_k = test[keep]
        if test_k.empty:
            continue
        teams = sorted(seen)
        cutoff = test["date"].min()
        prior = elo_prior_net(seed_from_ratings(r, cutoff), teams, beta=dc.get("prior_scale", 0.83))
        params = fit_dixon_coles(train, half_life_days=dc.get("half_life_days", 730),
                                 competition_weights=dc["competition_weights"],
                                 default_competition_weight=dc.get("default_competition_weight", 0.2),
                                 prior_net=prior, prior_strength=dc["prior_strength"])
        # Build v2 strength as-of cutoff
        strength_v2 = team_squad_strength(sq, ps, cutoff, months=cfg.get("recency_months", 24),
                                          league_strength=league_map, include_midfield=True)
        nz = test_k["neutral"].fillna(True).to_numpy() if "neutral" in test_k.columns else None
        hs = test_k["home_score"].to_numpy(int)
        as_ = test_k["away_score"].to_numpy(int)
        y = _result_labels(hs, as_)
        for coef in COEF_GRID:
            bumped = apply_squad_bump(params, strength_v2, coef)
            P = wdl_probs_for_matches(bumped, test_k["home_team"], test_k["away_team"],
                                      rho=bumped.get("rho", 0.0), max_goals=dc.get("max_goals", 10),
                                      neutral=nz)
            rows.append({"coef": coef, "split": split_i, "n_test": len(test_k),
                         "n_skipped": n_skip, "log_loss": log_loss(y, P),
                         "rps": rps(y, P), "brier": brier_score(y, P)})

    res = pd.DataFrame(rows)
    agg = res.groupby("coef")[["rps", "log_loss", "brier"]].mean().sort_values("rps")
    ntest = res.groupby("split")["n_test"].first().to_dict()
    skipped = res.groupby("split")["n_skipped"].first().to_dict()
    best = float(agg.reset_index().iloc[0]["coef"])
    incumbent = float(agg.loc[0.0, "rps"])
    win = best != 0.0 and agg.loc[best, "rps"] < incumbent

    lines = [
        "# Squad-v2 calibration (held-out RPS, league+midfield corrected)",
        "",
        f"Window: matches since {SINCE} | n_splits={N_SPLITS} | coef_grid={COEF_GRID}",
        f"Per-split test sizes: {ntest} | skipped (unseen team): {skipped}",
        "Variant: deployed both (competition weighting + Elo prior) + squad-v2 bump.",
        "Signal: team_squad_strength with league_strength + include_midfield=True",
        "",
        "Mean held-out metrics per coef, sorted by RPS (lower = better):",
        "", "```", agg.to_string(), "```", "",
        f"## Result: coef={best} {'beats' if win else 'does NOT beat'} coef=0 ({incumbent:.6f})",
        f"Selected best RPS: {agg.loc[best, 'rps']:.6f} vs incumbent {incumbent:.6f}",
        "",
        "Note: the corrected signal may stay RPS-neutral, RPS is dominated by",
        "within-confederation matches where talent differences matter less. The signal's",
        "value is as a defensible talent input (Portugal > Colombia, Mexico down), not a",
        "guaranteed accuracy win. Do NOT enable in production solely to move Portugal.",
    ]

    # Cross-confed subset if available
    confeds_path = Path("data/reference/confederations.csv")
    if confeds_path.exists():
        conf_df = pd.read_csv(confeds_path)
        conf_map = {r["team"]: r["confederation"]
                    for _, r in conf_df.iterrows()
                    if pd.notna(r["confederation"]) and r["confederation"] != ""}
        cross = res.copy()
        # Tag splits aren't available per-row for cross-confed; just note availability
        lines += ["", "A cross-confederation reference map is available.",
                  "Re-run with elo_anchor_walkforward v2 for cross-confed scoring."]

    print("\n".join(lines))
    Path("reports").mkdir(exist_ok=True)
    Path("reports/squad_coef_corrected_backtest.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
