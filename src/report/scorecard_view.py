# src/report/scorecard_view.py
"""Presenters for the forecast scorecard. Pure: frames in, frames and strings out."""
import pandas as pd

ROUND_LABELS = {"R32": "Round of 32", "R16": "Round of 16", "QF": "Quarter-finals",
                "SF": "Semi-finals", "F": "Final", "W": "Champion"}

# Surfaced in both the report and the dashboard so the two cannot drift apart.
WHAT_THIS_SCORES = (
    "What is being scored: the 72 group-stage predictions published on 11 June 2026, "
    "before the tournament began, and the tournament odds from the same run. The "
    "win/draw/loss probabilities came from the Elo baseline; the expected goals came "
    "from the corrected Dixon-Coles model. CatBoost is not scored here, because no "
    "CatBoost model was ever published for these fixtures, so the walk-forward figures "
    "on the Model page are a separate historical result and do not transfer."
)

HOW_TO_READ = (
    "Log-loss and RPS are lower-is-better. The two reference rows are there for scale: "
    "uniform assigns one third to every outcome, and always-home puts almost everything "
    "on the home side. A forecast that cannot beat both has no signal in it."
)


def headline(metrics, skill):
    """One-line summaries for the top of the page, as (label, value) pairs."""
    published = metrics[metrics["model"].str.startswith("published")].iloc[0]
    return [
        ("Matches scored", f"{int(published['n'])}"),
        ("Log-loss", f"{published['log_loss']:.3f}"),
        ("RPS", f"{published['rps']:.3f}"),
        ("Advantage over uniform", f"{skill['mean_advantage']:.3f}"),
    ]


def metrics_table(metrics):
    """Round the metric frame for display without changing the underlying numbers."""
    out = metrics.copy()
    for c in ("log_loss", "rps", "brier"):
        out[c] = out[c].round(4)
    return out


def goals_summary(goal_frame):
    """Readable rows for the expected-goals comparison."""
    g = goal_frame.iloc[0]
    direction = "fewer" if g["bias_per_side"] < 0 else "more"
    return pd.DataFrame([
        {"measure": "Mean goals predicted per match", "value": round(g["mean_predicted_goals"], 2)},
        {"measure": "Mean goals actually scored per match", "value": round(g["mean_actual_goals"], 2)},
        {"measure": "Mean absolute error per team per match", "value": round(g["mae_per_side"], 2)},
        {"measure": f"Bias per team per match ({direction} than actual)",
         "value": round(g["bias_per_side"], 2)},
    ])


def reliability_table(reliability):
    """Label the reach-probability buckets for display."""
    out = reliability.copy()
    out["mean_predicted"] = (out["mean_predicted"] * 100).round(1)
    out["observed_rate"] = (out["observed_rate"] * 100).round(1)
    return out.rename(columns={"bucket": "predicted probability",
                               "n": "cases",
                               "mean_predicted": "mean predicted %",
                               "observed_rate": "actually happened %"})


def per_match_table(joined, results):
    """Every scored match with its prediction and what happened, worst miss first.

    Sorted by the probability the model gave the outcome that occurred, so the matches
    it got most wrong are at the top. That ordering is deliberate: a scorecard that
    opens with its successes is marketing, not evaluation.
    """
    df = joined.merge(results[["match_id", "home_score", "away_score"]],
                      on="match_id", suffixes=("", "_r"))
    picked = df[["p_home", "p_draw", "p_away"]].to_numpy()
    idx = {"H": 0, "D": 1, "A": 2}
    df["p_actual"] = [row[idx[a]] for row, a in zip(picked, df["actual"])]
    df["score"] = (df["home_score"].astype(int).astype(str) + "-"
                   + df["away_score"].astype(int).astype(str))
    df["predicted"] = df[["p_home", "p_draw", "p_away"]].idxmax(axis=1).map(
        {"p_home": "home win", "p_draw": "draw", "p_away": "away win"})
    df["happened"] = df["actual"].map({"H": "home win", "D": "draw", "A": "away win"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    out = df[["date", "home_team", "away_team", "score", "predicted", "happened",
              "p_actual"]].copy()
    out["p_actual"] = (out["p_actual"] * 100).round(1)
    return out.sort_values("p_actual").reset_index(drop=True).rename(
        columns={"p_actual": "prob given to what happened (%)"})


def champion_sentence(check):
    """The title-odds outcome, stated with the caveat it needs.

    One tournament is one observation. A team given 25% wins about a quarter of the
    time, so this is neither evidence of skill nor evidence against it, and the
    sentence is written so it cannot be quoted as either.
    """
    if not check:
        return "No final result available."

    prob, rank = check["predicted_prob"], check["predicted_rank"]
    if rank < 1 or prob != prob:                      # absent from the odds, or NaN
        return (
            f"{check['champion']} won, beating {check['runner_up']} in the final. "
            f"The model's favorite was {check['favorite']} at "
            f"{check['favorite_prob']:.1%}. The winner does not appear in the title odds, "
            f"so there is no probability to report against them."
        )

    standing = ("the model's favorite" if rank == 1
                else f"the model's number {rank} pick of {check['n_teams']}")
    return (
        f"{check['champion']} won, and was {standing} at {prob:.1%}. "
        f"{check['runner_up']} were runners-up. That is one observation and it is worth "
        f"very little on its own: an outcome given {prob:.0%} happens about {prob:.0%} "
        f"of the time, so a single correct call is not evidence of skill. The 72 match "
        f"predictions above are."
    )
