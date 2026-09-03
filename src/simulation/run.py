from pathlib import Path


def apply_post_fit_corrections(params, matches, cfg=None, dc_cfg=None):
    """Apply both post-fit strength corrections, in the order that matters.

    Elo anchor first, since that is the validated cross-confederation correction,
    then the squad talent bump on top so its full weight survives. Doing it the other
    way round would dilute the talent weight, because the anchor would partly undo it.

    This is the single shared implementation, used by the simulation pipeline, the
    live runner for per-fixture expected goals, and the dashboard generator. It exists
    as one function because the runner once shipped uncorrected expected goals: it had
    its own copy of this logic and the copy fell behind.

    Both corrections are no-ops at a weight or coefficient of 0. Reads team_ratings,
    squad_coverage and player_stats from data/processed, so it is not fixture-tested;
    imports and file access are both lazy."""
    from src.config import load_params

    cfg = cfg if cfg is not None else load_params()
    dc_cfg = dc_cfg if dc_cfg is not None else cfg.get("models", {}).get("dixon_coles", {})

    elo_w = dc_cfg.get("elo_anchor_weight", 0.0)
    if elo_w and elo_w > 0:
        import pandas as pd
        from src.models.elo_prior import elo_prior_net_asof
        from src.simulation.params import apply_elo_anchor
        cutoff = cfg.get("states", {}).get("live_cutoff_date")
        fit_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
        tgt = dict(zip(fit_teams, elo_prior_net_asof(
            pd.read_parquet("data/processed/team_ratings.parquet"),
            fit_teams, cutoff, beta=dc_cfg.get("prior_scale", 0.83))))
        params = apply_elo_anchor(params, tgt, elo_w)
    squad_coef = dc_cfg.get("squad_coef", 0.0)
    if squad_coef and squad_coef > 0:
        import pandas as pd
        from src.features.squad_strength import team_squad_strength, club_league_multiplier
        from src.simulation.params import apply_squad_bump
        cutoff = cfg.get("states", {}).get("live_cutoff_date")
        squads = pd.read_csv("data/processed/squad_coverage.csv")
        pstats = pd.read_parquet("data/processed/player_stats.parquet")
        ls = club_league_multiplier(sorted(squads["club"].dropna().unique()))
        strength = team_squad_strength(squads, pstats, cutoff,
                                       months=cfg.get("recency_months", 24),
                                       league_strength=ls, include_midfield=True)
        params = apply_squad_bump(params, strength, squad_coef)
    return params


def run_simulation_pipeline(matches, fixtures, fmt, params=None, tournament="2026",
                            out_dir="data/processed", report_path="reports/simulation.md",
                            elo_prior=None, params_corrected=False):
    """Fit the goal model, simulate the tournament, persist the results and report.

    Fits Dixon-Coles on `matches`, reads the groups out of `fixtures`, runs the
    bracket the configured number of times (runs, seed and jitter all come from
    config/params.yaml), persists the simulation_results table, writes the report and
    returns the result frame.

    Not fixture-tested, since it triggers a real scipy fit and writes parquet files.
    Covered by an import smoke test and by the calibration scripts. scipy is imported
    lazily through fit_dixon_coles.

    When fitting (`params is None`) the fit uses the `models.dixon_coles` config:
    competition-tier weighting + (if `elo_prior` is supplied) the Elo-anchored
    shrinkage prior at `prior_strength`. `elo_prior` is a {team: target} dict (the
    centered strength target from `elo_prior.elo_prior_net`); without it the prior is
    off (prior_strength forced to 0) and the fit is weighting-only."""
    import numpy as np
    from src.config import load_params
    from src.models.dixon_coles import fit_dixon_coles
    from src.simulation.groups import parse_groups, groups_to_dict
    from src.simulation.montecarlo import run_simulation
    from src.ingest.run import persist_tables

    cfg = load_params()
    dc_cfg = cfg.get("models", {}).get("dixon_coles", {})
    if params is None:
        params = fit_dixon_coles(
            matches, half_life_days=dc_cfg.get("half_life_days", 730),
            competition_weights=dc_cfg.get("competition_weights"),
            default_competition_weight=dc_cfg.get("default_competition_weight", 0.2),
            prior_net=elo_prior,
            prior_strength=dc_cfg.get("prior_strength", 0.0) if elo_prior is not None else 0.0,
        )
    if not params_corrected:
        # apply the Elo anchor + squad bump (shared implementation; callers that
        # pre-correct, e.g. the live runner, pass params_corrected=True so the
        # corrections are never applied twice)
        params = apply_post_fit_corrections(params, matches, cfg=cfg, dc_cfg=dc_cfg)
    groups = groups_to_dict(parse_groups(fixtures, tournament=tournament))
    out, bracket = run_simulation(
        fmt, groups, params, fmt["seeding"],
        n_runs=cfg.get("sim_runs", 10000),
        rng=np.random.default_rng(cfg.get("random_seed", 42)),
        hosts=fmt.get("hosts", ()),
        jitter=cfg.get("sim_jitter", 0.0),
        rho=params.get("rho", 0.0),
        max_goals=dc_cfg.get("max_goals", 10),
        tournament=tournament,
        collect_bracket=True,
    )
    persist_tables({"simulation_results": out, "bracket_results": bracket}, out_dir=out_dir)
    write_simulation_report(out, path=report_path)
    return out


def write_simulation_report(results, path="reports/simulation.md", top=12):
    """Write the title-odds table to markdown, with intervals and a note on their meaning.

    The intervals are Monte-Carlo sampling error, meaning how much the number would
    move if the same model were run again with a different seed. They are not
    confidence in the model itself. Plain text output, no plotting."""
    from src.simulation.montecarlo import champion_odds
    odds = champion_odds(results, top=top)
    lines = ["# Tournament simulation: title odds", "",
             "| Team | P(win) | 95% CI |", "|---|---|---|"]
    for _, r in odds.iterrows():
        lines.append(f"| {r['team']} | {r['prob']:.3f} | [{r['ci_low']:.3f}, {r['ci_high']:.3f}] |")
    lines += ["",
              "> Monte-Carlo note: probabilities carry sampling error that shrinks as 1/sqrt(N). "
              "At N=10,000 the headline odds are precise to a few tenths of a percentage point, "
              "but rare events (longshot titles, exact qualification edge cases) carry meaningful "
              "*relative* error, the credible intervals above quantify it."]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(lines), encoding="utf-8")
    return path
