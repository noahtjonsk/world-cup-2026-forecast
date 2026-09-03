# src/evaluation/scorecard.py
"""Score the published 2026 forecast against what actually happened.

Everything here is pure. It takes the frozen predictions and the real results and
returns frames; it never refits a model and never touches a file.

Two things are worth stating up front, because they decide what these numbers mean.

The published win/draw/loss probabilities came from the Elo baseline, not CatBoost:
build_dashboard_data.py calls elo_baseline.predict_proba, and no CatBoost model was
ever serialized for these fixtures. The expected goals came from the corrected
Dixon-Coles model. So this scores what was actually published, which is the only thing
that can honestly be called a forecast.

Only the 72 group matches carry probabilities. No win/draw/loss numbers were ever
published for the knockouts, and extra time and shootouts would make the label
ambiguous anyway, so the knockouts feed the bracket comparison instead.
"""
import numpy as np
import pandas as pd

from src.evaluation.metrics import CLASSES, brier_score, calibration_table, log_loss, rps

# Wikipedia stage name -> the round code the simulation uses. The third-place playoff
# has no simulation counterpart and is deliberately absent.
STAGE_TO_ROUND = {
    "Round of 32": "R32",
    "Round of 16": "R16",
    "Quarterfinals": "QF",
    "Semifinals": "SF",
    "Final": "F",
}
# Increasing depth. "F" means reached the final; winning it is the separate round "W".
ROUND_ORDER = ["R32", "R16", "QF", "SF", "F"]
CHAMPION_ROUND = "W"


def outcome_labels(results):
    """H, D or A from the home team's perspective, for a results frame."""
    hs, as_ = results["home_score"].to_numpy(), results["away_score"].to_numpy()
    return np.where(hs > as_, "H", np.where(hs == as_, "D", "A"))


def join_predictions(predictions, results):
    """Inner-join the frozen predictions to the real results on match_id.

    Raises if any prediction is unmatched. A partial join would quietly score a subset
    and flatter whichever matches happened to survive.
    """
    res = results.copy()
    res["actual"] = outcome_labels(res)
    cols = ["match_id", "stage", "home_score", "away_score", "decided_by", "actual"]
    joined = predictions.merge(res[cols], on="match_id", how="left")
    missing = joined["actual"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} of {len(joined)} predictions have no result; "
            "refusing to score a subset"
        )
    return joined


def _probs(joined):
    return joined[["p_home", "p_draw", "p_away"]].to_numpy()


def wdl_metrics(joined):
    """Log-loss, RPS and Brier for the published probabilities and two reference
    predictors, as a tidy frame ordered best log-loss first.

    The references exist because a log-loss of 0.9 means nothing on its own. `uniform`
    assigns 1/3 to each outcome and is what you score by knowing nothing at all.
    `always home` puts almost everything on a home win, which on a neutral-venue
    tournament should be poor; if the model cannot beat these two, it has no signal.
    """
    y = joined["actual"].to_numpy()
    n = len(joined)
    published = _probs(joined)
    uniform = np.full((n, 3), 1 / 3)
    home = np.tile([0.90, 0.05, 0.05], (n, 1))

    rows = []
    for name, p in (("published (Elo baseline)", published),
                    ("uniform 1/3", uniform),
                    ("always home", home)):
        rows.append({
            "model": name, "n": n,
            "log_loss": log_loss(y, p, classes=CLASSES),
            "rps": rps(y, p, classes=CLASSES),
            "brier": brier_score(y, p, classes=CLASSES),
        })
    return pd.DataFrame(rows).sort_values("log_loss").reset_index(drop=True)


def goal_metrics(joined):
    """How the Dixon-Coles expected goals compared to the goals actually scored.

    `bias` is predicted minus actual, so a positive number means the model expected
    more goals than the tournament produced.
    """
    pred = np.concatenate([joined["exp_goals_home"], joined["exp_goals_away"]])
    actual = np.concatenate([joined["home_score"], joined["away_score"]]).astype(float)
    total_pred = joined["exp_goals_home"] + joined["exp_goals_away"]
    total_actual = (joined["home_score"] + joined["away_score"]).astype(float)
    return pd.DataFrame([{
        "n_matches": len(joined),
        "mae_per_side": float(np.mean(np.abs(pred - actual))),
        "bias_per_side": float(np.mean(pred - actual)),
        "mean_predicted_goals": float(total_pred.mean()),
        "mean_actual_goals": float(total_actual.mean()),
    }])


def calibration(joined, positive_class="H", n_bins=5):
    """Calibration of the published home-win probability, in `n_bins` buckets.

    Five bins rather than ten: 72 matches spread over ten buckets gives counts too
    small to read anything into.
    """
    return calibration_table(joined["actual"].to_numpy(), _probs(joined),
                             classes=CLASSES, positive_class=positive_class,
                             n_bins=n_bins)


def hit_rate(joined):
    """How often the single most likely outcome was the one that happened.

    Reported because it is the number people expect to see, not because it is the best
    measure. It throws away everything the probabilities say about confidence.
    """
    picked = np.array(CLASSES)[np.argmax(_probs(joined), axis=1)]
    return float(np.mean(picked == joined["actual"].to_numpy()))


def actual_rounds_reached(results):
    """{team: deepest round reached}, derived from who appeared in each knockout round.

    A team that played in the final reached the final, whether or not it won. Teams
    that never left the group stage are absent.
    """
    depth = {}
    for _, r in results.iterrows():
        rnd = STAGE_TO_ROUND.get(r["stage"])
        if rnd is None:
            continue                      # group stage, or the third-place playoff
        rank = ROUND_ORDER.index(rnd)
        for team in (r["home_team"], r["away_team"]):
            depth[team] = max(depth.get(team, -1), rank)
    return {t: ROUND_ORDER[i] for t, i in depth.items()}


def round_reach_comparison(simulation_results, results, tournament="2026"):
    """Predicted probability of reaching each round against whether the team did.

    `simulation_results` is the frozen per-team reach probabilities. The returned frame
    has one row per team and round, with `predicted` and `actual` (1 or 0), which is
    what a reliability check needs.
    """
    sim = simulation_results[simulation_results["tournament"] == tournament]
    reached = actual_rounds_reached(results)
    rows = []
    for _, r in sim.iterrows():
        rnd, team = r["round"], r["team"]
        if rnd not in ROUND_ORDER:
            continue
        got = reached.get(team)
        made_it = int(got is not None and ROUND_ORDER.index(got) >= ROUND_ORDER.index(rnd))
        rows.append({"team": team, "round": rnd,
                     "predicted": float(r["prob"]), "actual": made_it})
    return pd.DataFrame(rows)


def round_reliability(reach_frame, bins=(0.0, 0.1, 0.25, 0.5, 0.75, 1.0)):
    """Bucket the reach predictions and compare mean predicted against observed rate.

    This is the closest thing to a calibration check available for the tournament
    structure: across all teams and rounds, when the model said 25%, did about a
    quarter of those actually happen?
    """
    df = reach_frame.copy()
    df["bucket"] = pd.cut(df["predicted"], bins=list(bins), include_lowest=True)
    g = df.groupby("bucket", observed=True).agg(
        n=("actual", "size"),
        mean_predicted=("predicted", "mean"),
        observed_rate=("actual", "mean"),
    ).reset_index()
    g["bucket"] = g["bucket"].astype(str)
    return g


def champion_check(simulation_results, results, tournament="2026"):
    """What the model gave the team that actually won, and where it ranked them.

    One observation. It cannot support a claim about skill in either direction, which
    is why this returns the rank alongside the probability rather than a verdict.
    """
    final = results[results["stage"] == "Final"]
    if final.empty:
        return None
    f = final.iloc[0]
    winner = (f["home_team"] if f["home_score"] > f["away_score"] else f["away_team"])
    runner_up = (f["away_team"] if f["home_score"] > f["away_score"] else f["home_team"])

    champ = simulation_results[(simulation_results["tournament"] == tournament)
                               & (simulation_results["round"] == CHAMPION_ROUND)]
    champ = champ.sort_values("prob", ascending=False).reset_index(drop=True)
    row = champ[champ["team"] == winner]
    return {
        "champion": winner,
        "runner_up": runner_up,
        "predicted_prob": float(row["prob"].iloc[0]) if len(row) else float("nan"),
        "predicted_rank": int(row.index[0]) + 1 if len(row) else -1,
        "n_teams": len(champ),
        "favorite": champ["team"].iloc[0],
        "favorite_prob": float(champ["prob"].iloc[0]),
    }


def skill_interval(joined, n_boot=10000, seed=42):
    """Bootstrap the per-match log-loss advantage over the uniform predictor.

    72 matches is a modest sample, so the headline gap needs an interval around it
    before anyone leans on it. Resamples matches with replacement and returns the mean
    advantage with a 95% percentile interval. If the interval includes zero, the
    forecast is not distinguishable from knowing nothing, whatever the point estimate
    happens to look like.

    Positive numbers mean the published forecast did better.
    """
    y = joined["actual"].to_numpy()
    idx = {c: i for i, c in enumerate(CLASSES)}
    truth = np.array([idx[v] for v in y])
    p = np.clip(_probs(joined), 1e-15, 1.0)

    per_match_published = -np.log(p[np.arange(len(truth)), truth])
    per_match_uniform = np.full(len(truth), -np.log(1 / 3))
    diff = per_match_uniform - per_match_published        # >0 means published is better

    rng = np.random.default_rng(seed)
    boot = rng.choice(diff, size=(n_boot, len(diff)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "mean_advantage": float(diff.mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n": int(len(diff)),
        "beats_uniform": bool(lo > 0),
    }


def outcome_mix(joined):
    """Mean predicted probability against observed frequency, per outcome class.

    Shows which way the forecast leaned overall, separately from whether it got
    individual matches right.
    """
    rows = []
    for cls, col in zip(CLASSES, ("p_home", "p_draw", "p_away")):
        rows.append({
            "outcome": {"H": "home win", "D": "draw", "A": "away win"}[cls],
            "mean_predicted": float(joined[col].mean()),
            "observed": float((joined["actual"] == cls).mean()),
        })
    return pd.DataFrame(rows)
