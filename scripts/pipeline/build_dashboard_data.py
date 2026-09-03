"""Build the two tables the dashboard needs that the simulation pipeline does not persist.

Produces matchup_features and match_predictions for the 72 upcoming fixtures:
recompute Elo, take an as-of-kickoff feature snapshot per fixture, run the W/D/L
model, then the corrected Dixon-Coles model for expected goals. It deliberately
stops short of the 10,000-run Monte-Carlo, since the existing simulation results
already drive the Tournament page.

The same post-fit corrections the bracket uses are applied here, so per-fixture
expected goals always agree with the tournament odds. Predicted elevens are
re-keyed from fixture_id to match_id so the squad features populate.

    python scripts/pipeline/build_dashboard_data.py"""
import numpy as np
import pandas as pd

from src.config import load_params
from src.schema import CANON_MATCH_COLS
from src.utils.io import read_parquet
from src.utils.ids import make_match_id
from src.states.elo_update import recompute_elo, seed_from_ratings
from src.states.predict import upcoming_fixtures, assemble_predictions
from src.features.snapshot import build_feature_table
from src.models.dixon_coles import fit_dixon_coles
from src.models.goals import score_matrix, expected_goals_from_matrix
from src.models.elo_baseline import predict_proba as elo_predict
from src.simulation.match import match_lambdas
from src.simulation.format import load_tournament_format
from src.ingest.run import persist_tables


def main(out_dir="data/processed"):
    cfg = load_params()
    scfg, mcfg = cfg.get("states", {}), cfg.get("models", {})
    dc_cfg = mcfg.get("dixon_coles", {})
    as_of = scfg.get("live_cutoff_date")

    matches = read_parquet("data/processed/matches.parquet")
    fixtures = read_parquet("data/processed/fixtures.parquet")
    rt = read_parquet("data/processed/team_ratings.parquet")
    ps = read_parquet("data/processed/player_stats.parquet")
    ts = read_parquet("data/processed/team_style.parquet")
    lineups = read_parquet("data/processed/lineups.parquet")
    fmt = load_tournament_format("2026")
    hosts = fmt.get("hosts", ())
    seed = seed_from_ratings(rt, as_of)

    # 1. Elo recompute: only results on/after the cutoff (the seed already prices in
    # all earlier history; replaying it double-counts, pre-tournament this is empty)
    live_elo = recompute_elo(matches, seed,
                             k_by_competition=scfg.get("competition_weights", {}),
                             default_k=scfg.get("k_base", 60.0), since=as_of)
    ratings = rt if live_elo.empty else pd.concat([rt, live_elo], ignore_index=True)

    # 2. upcoming fixtures -> unplayed match rows -> leakage-safe feature snapshots
    up = upcoming_fixtures(fixtures, as_of)
    fixt_matches = pd.DataFrame({
        "match_id": [make_match_id(pd.Timestamp(d), h, a)
                     for d, h, a in zip(up["date"], up["home_team"], up["away_team"])],
        "date": pd.to_datetime(up["date"]).to_numpy(),
        "competition": "World Cup", "season": "2026",
        "home_team": up["home_team"].to_numpy(), "away_team": up["away_team"].to_numpy(),
        "home_score": np.nan, "away_score": np.nan,
        "stage": up["stage"].to_numpy(), "neutral": up["neutral"].to_numpy(),
        "source": "fixture",
    })[CANON_MATCH_COLS]
    all_matches = pd.concat([matches, fixt_matches], ignore_index=True)

    # re-key predicted XIs (fixture_id-keyed) to match_id so squad features populate
    fx = fixtures.copy()
    fx["match_id"] = [make_match_id(pd.Timestamp(d), h, a)
                      for d, h, a in zip(fx["date"], fx["home_team"], fx["away_team"])]
    id_map = dict(zip(fx["fixture_id"], fx["match_id"]))
    lu = lineups.copy()
    lu["match_id"] = lu["fixture_id"].map(id_map)
    lu = lu.dropna(subset=["match_id"])

    feats = build_feature_table(
        list(fixt_matches["match_id"]), all_matches, ratings, ps, ts, lu,
        months=cfg.get("recency_months", 24), host_teams=hosts, snapshot_date=as_of,
    )

    # 3. W/D/L via the Elo baseline (no CatBoost dependency)
    wdl = elo_predict(feats["elo_diff"].to_numpy(), feats["neutral"].to_numpy(),
                      home_adv=mcfg.get("elo_home_advantage", 65.0),
                      draw_base=mcfg.get("elo_draw_base", 0.30))

    # 4. refit Dixon-Coles -> expected goals per upcoming fixture
    elo_prior = None
    if dc_cfg.get("prior_strength", 0.0) > 0:
        from src.models.elo_prior import elo_prior_net
        fit_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        elo_prior = dict(zip(fit_teams, elo_prior_net(
            seed, fit_teams, beta=dc_cfg.get("prior_scale", 0.83))))
    params = fit_dixon_coles(
        matches, half_life_days=dc_cfg.get("half_life_days", 730),
        competition_weights=dc_cfg.get("competition_weights"),
        default_competition_weight=dc_cfg.get("default_competition_weight", 0.2),
        prior_net=elo_prior,
        prior_strength=dc_cfg.get("prior_strength", 0.0) if elo_prior is not None else 0.0,
    )
    for t in set(up["home_team"]) | set(up["away_team"]):
        params["attack"].setdefault(t, 0.0)
        params["defence"].setdefault(t, 0.0)

    # Apply the SAME post-fit corrections the bracket sim uses (shared implementation:
    # Elo anchor first, then the corrected squad bump) so the Match page's expected
    # goals agree with the strengths behind the title odds.
    from src.simulation.run import apply_post_fit_corrections
    params = apply_post_fit_corrections(params, matches, cfg=cfg, dc_cfg=dc_cfg)

    max_goals, rho = dc_cfg.get("max_goals", 10), params.get("rho", 0.0)
    exp_goals = []
    for _, f in up.iterrows():
        lam_h, lam_a = match_lambdas(params, f["home_team"], f["away_team"], hosts=hosts)
        exp_goals.append(expected_goals_from_matrix(
            score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)))

    preds = assemble_predictions(up, wdl, exp_goals, snapshot_date=as_of)

    persist_tables({"matchup_features": feats, "match_predictions": preds}, out_dir=out_dir)
    cov = feats["xi_quality_home"].notna().mean()
    print(f"persisted matchup_features ({len(feats)} rows, xi_quality coverage "
          f"{cov:.0%}) + match_predictions ({len(preds)} rows) to {out_dir}")
    print(preds[["home_team", "away_team", "p_home", "p_draw", "p_away",
                 "exp_goals_home", "exp_goals_away"]].head(8).to_string(index=False))


if __name__ == "__main__":
    main()
