COARSE_UPDATE_NOTE = (
    "Live updates are COARSE: Elo + lineups + summary stats only, not event-level. "
    "Forecasts refresh per result; the CatBoost W/D/L model is reused (periodic "
    "out-of-band refit), while the Dixon-Coles goal model and the Monte-Carlo "
    "simulation are rerun each cycle."
)

def run_live_update(matches, fixtures, seed_ratings, fmt, player_stats, team_style,
                    lineups=None, wdl_model=None, as_of=None, base_ratings=None,
                    out_dir="data/processed"):
    """Run one after-match update cycle and return the match_predictions frame.

    Steps:
      1. recompute Elo forward from the seed, which is idempotent, and append to the
         existing series;
      2. select the fixtures still to come and rebuild their feature snapshots as of
         kickoff (without lineups the squad features come through as NaN, which is
         what makes this a coarse update);
      3. rerun W/D/L, reusing `wdl_model` if given and falling back to the Elo
         baseline otherwise;
      4. refit Dixon-Coles on the updated results for expected goals per fixture;
      5. rerun the Monte-Carlo simulation;
      6. persist ratings, matchup_features and match_predictions, with
         simulation_results written by the simulation pipeline.

    This is an orchestrator: the work lives in the modules it calls. Not
    fixture-tested, since it touches parquet files, a scipy fit and CatBoost. Covered
    by an import smoke test and by scripts/calibration/verify_live_update.py. scipy
    and catboost are both imported lazily by the functions that need them."""
    import numpy as np
    import pandas as pd
    from src.config import load_params
    from src.schema import CANON_MATCH_COLS
    from src.states.elo_update import recompute_elo
    from src.states.predict import upcoming_fixtures, assemble_predictions
    from src.features.snapshot import build_feature_table
    from src.models.dixon_coles import fit_dixon_coles
    from src.models.goals import score_matrix, expected_goals_from_matrix
    from src.models.elo_baseline import predict_proba as elo_predict
    from src.models.wdl import feature_columns, predict_proba as cat_predict
    from src.simulation.match import match_lambdas
    from src.simulation.run import run_simulation_pipeline
    from src.ingest.run import persist_tables
    from src.utils.ids import make_match_id

    cfg = load_params()
    scfg = cfg.get("states", {})
    mcfg = cfg.get("models", {})
    as_of = as_of or scfg.get("live_cutoff_date")
    hosts = fmt.get("hosts", ())

    # 1. recompute Elo (idempotent) and combine with the prior series. since=as_of:
    # the seed already prices in all pre-cutoff history, replay ONLY new results
    # (replaying the full table re-applied 150 years of matches on current ratings).
    live_elo = recompute_elo(matches, seed_ratings,
                             k_by_competition=scfg.get("competition_weights", {}),
                             default_k=scfg.get("k_base", 60.0), since=as_of)
    if base_ratings is None:
        ratings = live_elo
    elif live_elo.empty:
        ratings = base_ratings
    else:
        ratings = pd.concat([base_ratings, live_elo], ignore_index=True)

    # 2. upcoming fixtures -> as unplayed matches rows -> rebuild snapshots
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
    feats = build_feature_table(
        list(fixt_matches["match_id"]), all_matches, ratings, player_stats, team_style,
        lineups if lineups is not None else pd.DataFrame(),
        months=cfg.get("recency_months", 24), host_teams=hosts, snapshot_date=as_of,
    )

    # 3. W/D/L: reuse trained CatBoost, else the Elo baseline
    if wdl_model is not None:
        wdl = cat_predict(wdl_model, feats[feature_columns(feats.columns)])
    else:
        wdl = elo_predict(feats["elo_diff"].to_numpy(), feats["neutral"].to_numpy(),
                          home_adv=mcfg.get("elo_home_advantage", 65.0),
                          draw_base=mcfg.get("elo_draw_base", 0.30))

    # 4. refit Dixon-Coles -> expected goals per upcoming fixture
    dc_cfg = mcfg.get("dixon_coles", {})
    # Elo-anchored prior (leakage-safe: seed_ratings is the pre-cutoff Elo seed). Pass as a
    # {team: target} dict so the fit aligns it by name regardless of its internal team set.
    elo_prior = None
    if dc_cfg.get("prior_strength", 0.0) > 0:
        from src.models.elo_prior import elo_prior_net
        fit_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        elo_prior = dict(zip(fit_teams, elo_prior_net(
            seed_ratings, fit_teams, beta=dc_cfg.get("prior_scale", 0.35))))
    params = fit_dixon_coles(
        matches, half_life_days=dc_cfg.get("half_life_days", 730),
        competition_weights=dc_cfg.get("competition_weights"),
        default_competition_weight=dc_cfg.get("default_competition_weight", 0.2),
        prior_net=elo_prior,
        prior_strength=dc_cfg.get("prior_strength", 0.0) if elo_prior is not None else 0.0,
    )
    for t in set(up["home_team"]) | set(up["away_team"]):    # unseen teams -> league average
        params["attack"].setdefault(t, 0.0)
        params["defence"].setdefault(t, 0.0)
    # apply the SAME post-fit corrections the bracket sim uses (Elo anchor + squad
    # bump) so per-fixture expected goals agree with the strengths behind the odds
    from src.simulation.run import apply_post_fit_corrections
    params = apply_post_fit_corrections(params, matches, cfg=cfg, dc_cfg=dc_cfg)
    max_goals, rho = dc_cfg.get("max_goals", 10), params.get("rho", 0.0)
    exp_goals = []
    for _, f in up.iterrows():
        lam_h, lam_a = match_lambdas(params, f["home_team"], f["away_team"], hosts=hosts)
        exp_goals.append(expected_goals_from_matrix(score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)))

    preds = assemble_predictions(up, wdl, exp_goals, snapshot_date=as_of)

    # 5. rerun the Monte-Carlo sim (reuse the freshly fit + corrected params;
    # params_corrected=True so the pipeline does not anchor/bump a second time)
    run_simulation_pipeline(matches, fixtures, fmt, params=params, tournament="2026",
                            out_dir=out_dir, params_corrected=True)

    # 6. persist the refreshed tables
    persist_tables({"ratings": ratings, "matchup_features": feats,
                    "match_predictions": preds}, out_dir=out_dir)
    return preds
