"""Does the squad bump add anything on top of the Elo anchor?

An ablation over both corrections at once: Elo-anchor weight (0 for a plain
Dixon-Coles fit, 0.7 for the deployed value) crossed with squad coefficient,
scored only on cross-confederation matches, where a gap between talent and Elo is
actually visible. Judged on RPS, never on title odds.

Caveat: historical squads are proxied by 2026 rosters. Player quality is still
measured as of the match date, so nothing leaks, but the roster itself gets less
accurate the further back the match sits.
Writes reports/squad_xconf_backtest.md."""
from pathlib import Path
import pandas as pd
from src.config import load_params
from src.features.squad_strength import club_league_multiplier
from src.evaluation.goal_backtest import squad_xconf_walkforward

COEF_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
ELO_W = (0.0, 0.7)          # 0.0 = plain DC; 0.7 = production Elo-anchored baseline
N_SPLITS = 4
SINCE = "2014-01-01"


def main():
    m = pd.read_parquet("data/processed/matches.parquet")
    r = pd.read_parquet("data/processed/team_ratings.parquet")
    ps = pd.read_parquet("data/processed/player_stats.parquet")
    sq = pd.read_csv("data/processed/squad_coverage.csv")
    c = pd.read_csv("data/reference/confederations.csv")
    confeds = dict(zip(c["team"], c["confederation"]))
    ls = club_league_multiplier(sorted(sq["club"].dropna().unique()))
    cfg = load_params(); dc = cfg["models"]["dixon_coles"]

    res = squad_xconf_walkforward(
        m, r, confeds, sq, ps, league_strength=ls, n_splits=N_SPLITS,
        coef_grid=COEF_GRID, elo_anchor_weights=ELO_W,
        competition_weights=dc["competition_weights"],
        default_competition_weight=dc.get("default_competition_weight", 0.2),
        prior_strength=dc["prior_strength"], beta=dc.get("prior_scale", 0.83),
        since=SINCE, months=cfg.get("recency_months", 24))

    agg = res.groupby(["elo_w", "coef"])[["rps", "log_loss", "brier"]].mean()
    ncross = res.groupby("split")["n_cross"].first().to_dict()
    minc = min(ncross.values())
    prod = float(agg.loc[(0.7, 0.0), "rps"])                 # production baseline: Elo anchor, no squad
    on_elo = agg.loc[0.7].sort_values("rps")
    best_coef_on_elo = float(on_elo.index[0])
    helps_on_elo = best_coef_on_elo != 0.0 and float(on_elo.iloc[0]["rps"]) < prod
    plain = float(agg.loc[(0.0, 0.0), "rps"])
    plain_best = float(agg.loc[0.0].sort_values("rps").index[0])

    lines = [
        "# Cross-confederation calibration of the corrected squad signal",
        "",
        f"Window: matches since {SINCE} | n_splits={N_SPLITS} | coef_grid={COEF_GRID} | elo_anchor_weights={ELO_W}",
        f"Per-split cross-confed test sizes: {ncross} | min n_cross/split: {minc}",
        "Signal: team_squad_strength(league_strength=..., include_midfield=True), as-of test-window start.",
        "Roster-proxy caveat: 2026 rosters proxy historical squads (quality leakage-safe as-of date).",
        "",
        "Mean cross-confed held-out metrics per (elo_w, coef):",
        "", "```", agg.to_string(), "```", "",
        "## Reading",
        f"- Plain DC (elo_w=0.0): squad coef={plain_best} best vs coef=0 ({plain:.6f}) "
        "Does talent help in isolation?",
        f"- **Production (elo_w=0.7, Elo anchor on): baseline RPS={prod:.6f}; best squad coef on top = "
        f"{best_coef_on_elo}** → squad **{'ADDS accuracy' if helps_on_elo else 'does NOT add accuracy'}** "
        "on top of the Elo anchor.",
        "",
        f"## Verdict: squad on top of Elo anchor {'beats' if helps_on_elo else 'does NOT beat'} the production "
        f"baseline (smallest cross-confederation sample per split = {minc}). Judged on cross-confederation RPS, never on title odds.",
    ]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/squad_xconf_backtest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
