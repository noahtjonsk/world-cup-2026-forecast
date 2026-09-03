import numpy as np
import pandas as pd
from src.models.goals import score_matrix, matrix_to_wdl
from src.models.dixon_coles import team_lambdas
from src.evaluation.metrics import log_loss, rps, brier_score
from src.evaluation.walkforward import time_splits

def wdl_probs_for_matches(params, home, away, rho=0.0, max_goals=10, neutral=None):
    """(n,3) [H,D,A] probabilities for each (home, away) pair from fitted DC params.
    Pure: composes team_lambdas (unseen teams -> 0) + score_matrix + matrix_to_wdl.
    If neutral[i] is True, home_adv is dropped for that row."""
    home = list(home); away = list(away)
    neutral = [False] * len(home) if neutral is None else list(neutral)
    atk, dfc = params["attack"], params["defence"]
    rows = []
    for h, a, nz in zip(home, away, neutral):
        p = {"attack": {**atk, h: atk.get(h, 0.0), a: atk.get(a, 0.0)},
             "defence": {**dfc, h: dfc.get(h, 0.0), a: dfc.get(a, 0.0)},
             "home_adv": 0.0 if nz else params["home_adv"]}
        lam_h, lam_a = team_lambdas(p, h, a)
        rows.append(matrix_to_wdl(score_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)))
    return np.asarray(rows, dtype=float)

def _result_labels(hs, as_):
    return np.where(hs > as_, "H", np.where(hs < as_, "A", "D"))

def goalmodel_walkforward(matches, ratings, n_splits=4, lam_grid=(0.25, 0.5, 1.0, 2.0),
                          competition_weights=None, default_competition_weight=0.2,
                          beta=0.35, max_goals=10, half_life_days=730,
                          since="2014-01-01", variants=None):
    """Walk-forward backtest of the goal model over a recent window (`since`).

    Four variants: the plain fit, competition weighting alone, the Elo prior alone at
    each lambda above 0, and both together. Per split it fits on the training matches,
    builds the Elo prior as of the start of the test window so nothing leaks, scores
    the test W/D/L, and skips test rows whose home or away team never appears in
    training.

    Returns tidy rows of [variant, lam, split, n_test, n_skipped, log_loss, rps,
    brier]. Pass `variants` to restrict the run, for instance {"both"} for a focused
    confirm grid; the default scores all four.

    Not unit-tested, since it needs a real scipy fit. Covered by an import smoke test
    and by the calibration scripts that call it. scipy is imported lazily through
    fit_dixon_coles, so importing this module does not require it."""
    from src.models.dixon_coles import fit_dixon_coles
    from src.states.elo_update import seed_from_ratings
    from src.models.elo_prior import elo_prior_net

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    rows = []
    for split_i, (tr, te) in enumerate(time_splits(df, n_splits=n_splits)):
        train, test = df.loc[tr], df.loc[te]
        seen = set(train["home_team"]) | set(train["away_team"])
        keep = test["home_team"].isin(seen) & test["away_team"].isin(seen)
        n_skip = int((~keep).sum())
        test_k = test[keep]
        if test_k.empty:
            continue
        teams = sorted(seen)
        cutoff = test["date"].min()                                 # test-window start
        prior = elo_prior_net(seed_from_ratings(ratings, cutoff), teams, beta=beta)
        nz = test_k["neutral"].fillna(True).to_numpy() if "neutral" in test_k.columns else None
        hs = test_k["home_score"].to_numpy(int); as_ = test_k["away_score"].to_numpy(int)
        y = _result_labels(hs, as_)

        def _score(variant, lam, fit_kwargs):
            if variants is not None and variant not in variants:
                return                                          # skip fit too (expensive)
            params = fit_dixon_coles(train, half_life_days=half_life_days, **fit_kwargs)
            P = wdl_probs_for_matches(params, test_k["home_team"], test_k["away_team"],
                                      rho=params.get("rho", 0.0), max_goals=max_goals, neutral=nz)
            rows.append({"variant": variant, "lam": lam, "split": split_i,
                         "n_test": len(test_k), "n_skipped": n_skip,
                         "log_loss": log_loss(y, P), "rps": rps(y, P), "brier": brier_score(y, P)})

        cw = competition_weights
        _score("baseline", 0.0, {})
        if cw is not None:
            _score("weight", 0.0, {"competition_weights": cw,
                                   "default_competition_weight": default_competition_weight})
        for lam in lam_grid:
            if lam <= 0:
                continue
            _score("prior", lam, {"prior_net": prior, "prior_strength": lam})
            if cw is not None:
                _score("both", lam, {"competition_weights": cw,
                                     "default_competition_weight": default_competition_weight,
                                     "prior_net": prior, "prior_strength": lam})
    return pd.DataFrame(rows)


def squad_coef_walkforward(matches, ratings, squads, player_stats, n_splits=3,
                           coef_grid=(0.0, 0.25, 0.5, 1.0, 2.0),
                           competition_weights=None, default_competition_weight=0.2,
                           prior_strength=1.5, beta=0.83, max_goals=10,
                           half_life_days=730, since="2023-01-01", months=24):
    """Walk-forward calibration of the squad-strength coefficient on recent matches.

    Per split: fit Dixon-Coles with the deployed configuration (competition weighting
    plus the Elo prior at `prior_strength`), build team_squad_strength as of the start
    of the test window so no future information leaks in, then for each coefficient
    apply the squad bump and score the held-out W/D/L. A coefficient of 0 leaves the
    fit untouched. Test rows whose team never appears in training are skipped.

    Returns tidy rows of [coef, split, n_test, n_skipped, log_loss, rps, brier]. The
    coefficient is chosen on held-out RPS, never on title odds. `beta` defaults to
    0.83, the locked prior scale, rather than goalmodel_walkforward's 0.35.

    scipy is imported lazily through fit_dixon_coles, so importing this module does
    not require it."""
    from src.models.dixon_coles import fit_dixon_coles
    from src.states.elo_update import seed_from_ratings
    from src.models.elo_prior import elo_prior_net
    from src.features.squad_strength import team_squad_strength
    from src.simulation.params import apply_squad_bump

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    rows = []
    for split_i, (tr, te) in enumerate(time_splits(df, n_splits=n_splits)):
        train, test = df.loc[tr], df.loc[te]
        seen = set(train["home_team"]) | set(train["away_team"])
        keep = test["home_team"].isin(seen) & test["away_team"].isin(seen)
        n_skip = int((~keep).sum())
        test_k = test[keep]
        if test_k.empty:
            continue
        teams = sorted(seen)
        cutoff = test["date"].min()
        prior = elo_prior_net(seed_from_ratings(ratings, cutoff), teams, beta=beta)
        params = fit_dixon_coles(train, half_life_days=half_life_days,
                                 competition_weights=competition_weights,
                                 default_competition_weight=default_competition_weight,
                                 prior_net=prior, prior_strength=prior_strength)
        strength = team_squad_strength(squads, player_stats, cutoff, months=months)
        nz = test_k["neutral"].fillna(True).to_numpy() if "neutral" in test_k.columns else None
        hs = test_k["home_score"].to_numpy(int); as_ = test_k["away_score"].to_numpy(int)
        y = _result_labels(hs, as_)
        for coef in coef_grid:
            bumped = apply_squad_bump(params, strength, coef)
            P = wdl_probs_for_matches(bumped, test_k["home_team"], test_k["away_team"],
                                      rho=bumped.get("rho", 0.0), max_goals=max_goals, neutral=nz)
            rows.append({"coef": coef, "split": split_i, "n_test": len(test_k),
                         "n_skipped": n_skip, "log_loss": log_loss(y, P),
                         "rps": rps(y, P), "brier": brier_score(y, P)})
    return pd.DataFrame(rows)


def elo_anchor_walkforward(matches, ratings, confeds, n_splits=4,
                           weight_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
                           competition_weights=None, default_competition_weight=0.2,
                           prior_strength=1.5, beta=0.83, max_goals=10,
                           half_life_days=730, since="2014-01-01"):
    """Walk-forward calibration of the Elo-anchor weight.

    Scored only on test matches where home and away come from different
    confederations, since that is where the gap between goal-model strength and Elo
    shows up. `confeds` maps team to confederation.

    Per split: fit Dixon-Coles with the deployed configuration, build the Elo target
    as of the start of the test window, then for each weight apply the anchor and
    score the cross-confederation subset. A weight of 0 leaves the fit untouched.

    Returns tidy rows of [weight, split, n_cross, n_skipped, rps, log_loss, brier].
    The weight is chosen on held-out RPS, never on title odds. scipy is imported
    lazily, so importing this module does not require it."""
    from src.models.dixon_coles import fit_dixon_coles
    from src.states.elo_update import seed_from_ratings
    from src.models.elo_prior import elo_prior_net
    from src.simulation.params import apply_elo_anchor

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    rows = []
    for split_i, (tr, te) in enumerate(time_splits(df, n_splits=n_splits)):
        train, test = df.loc[tr], df.loc[te]
        seen = set(train["home_team"]) | set(train["away_team"])
        # keep test rows: both teams seen in train AND both mapped to confederations AND cross-confed
        def _cross(h, a):
            return h in confeds and a in confeds and confeds[h] != confeds[a]
        keep = test.apply(lambda r: (r["home_team"] in seen and r["away_team"] in seen
                                     and _cross(r["home_team"], r["away_team"])), axis=1)
        n_skip = int((~keep).sum())
        test_k = test[keep]
        if test_k.empty:
            continue
        teams = sorted(seen)
        cutoff = test["date"].min()
        prior = elo_prior_net(seed_from_ratings(ratings, cutoff), teams, beta=beta)
        params = fit_dixon_coles(train, half_life_days=half_life_days,
                                 competition_weights=competition_weights,
                                 default_competition_weight=default_competition_weight,
                                 prior_net=prior, prior_strength=prior_strength)
        elo_target = dict(zip(teams, prior))                 # {team: centered target}
        nz = test_k["neutral"].fillna(True).to_numpy() if "neutral" in test_k.columns else None
        hs = test_k["home_score"].to_numpy(int); as_ = test_k["away_score"].to_numpy(int)
        y = _result_labels(hs, as_)
        for w in weight_grid:
            p = apply_elo_anchor(params, elo_target, w)
            P = wdl_probs_for_matches(p, test_k["home_team"], test_k["away_team"],
                                      rho=p.get("rho", 0.0), max_goals=max_goals, neutral=nz)
            rows.append({"weight": w, "split": split_i, "n_cross": len(test_k),
                         "n_skipped": n_skip, "log_loss": log_loss(y, P),
                         "rps": rps(y, P), "brier": brier_score(y, P)})
    return pd.DataFrame(rows)


def squad_xconf_walkforward(matches, ratings, confeds, squads, player_stats,
                            league_strength=None, n_splits=4,
                            coef_grid=(0.0, 0.25, 0.5, 1.0, 2.0),
                            elo_anchor_weights=(0.0, 0.7),
                            competition_weights=None, default_competition_weight=0.2,
                            prior_strength=1.5, beta=0.83, max_goals=10,
                            half_life_days=730, since="2014-01-01", months=24):
    """Walk-forward ablation of the corrected squad coefficient against the Elo anchor.

    Scored only on cross-confederation test matches, where a gap between talent and
    Elo is visible. For each anchor weight (0.0 for a plain Dixon-Coles baseline, 0.7
    for the deployed one) it sweeps the squad coefficient, which answers two questions
    at once: whether the corrected squad signal helps on its own, and whether it adds
    anything on top of the anchor. The second is the one that matters in production.

    The corrected squad strength, counting midfielders and weighting by league, is
    built as of the start of the test window, so player quality never sees the future.
    The roster itself is the 2026 squad used as a proxy, so it gets less accurate the
    further back the match sits.

    Returns tidy rows of [elo_w, coef, split, n_cross, n_skipped, rps, log_loss,
    brier]. Chosen on cross-confederation RPS, never on title odds. scipy is imported
    lazily, so importing this module does not require it."""
    from src.models.dixon_coles import fit_dixon_coles
    from src.states.elo_update import seed_from_ratings
    from src.models.elo_prior import elo_prior_net
    from src.features.squad_strength import team_squad_strength
    from src.simulation.params import apply_squad_bump, apply_elo_anchor

    df = matches.dropna(subset=["home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    if since is not None:
        df = df[df["date"] >= pd.Timestamp(since)]
    df = df.sort_values("date", kind="mergesort").reset_index(drop=True)
    rows = []
    for split_i, (tr, te) in enumerate(time_splits(df, n_splits=n_splits)):
        train, test = df.loc[tr], df.loc[te]
        seen = set(train["home_team"]) | set(train["away_team"])
        def _cross(h, a):
            return h in confeds and a in confeds and confeds[h] != confeds[a]
        keep = test.apply(lambda r: (r["home_team"] in seen and r["away_team"] in seen
                                     and _cross(r["home_team"], r["away_team"])), axis=1)
        n_skip = int((~keep).sum())
        test_k = test[keep]
        if test_k.empty:
            continue
        teams = sorted(seen)
        cutoff = test["date"].min()
        prior = elo_prior_net(seed_from_ratings(ratings, cutoff), teams, beta=beta)
        params = fit_dixon_coles(train, half_life_days=half_life_days,
                                 competition_weights=competition_weights,
                                 default_competition_weight=default_competition_weight,
                                 prior_net=prior, prior_strength=prior_strength)
        elo_target = dict(zip(teams, prior))
        strength = team_squad_strength(squads, player_stats, cutoff, months=months,
                                       league_strength=league_strength, include_midfield=True)
        nz = test_k["neutral"].fillna(True).to_numpy() if "neutral" in test_k.columns else None
        hs = test_k["home_score"].to_numpy(int); as_ = test_k["away_score"].to_numpy(int)
        y = _result_labels(hs, as_)
        for ew in elo_anchor_weights:
            base = apply_elo_anchor(params, elo_target, ew)
            for coef in coef_grid:
                p = apply_squad_bump(base, strength, coef)
                P = wdl_probs_for_matches(p, test_k["home_team"], test_k["away_team"],
                                          rho=p.get("rho", 0.0), max_goals=max_goals, neutral=nz)
                rows.append({"elo_w": ew, "coef": coef, "split": split_i,
                             "n_cross": len(test_k), "n_skipped": n_skip,
                             "log_loss": log_loss(y, P), "rps": rps(y, P), "brier": brier_score(y, P)})
    return pd.DataFrame(rows)
