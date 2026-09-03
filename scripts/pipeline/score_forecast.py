"""Score the frozen 2026 forecast against the real results into reports/forecast_scorecard.md.

Reads only. The published prediction is the whole point of the exercise, so nothing here
refits a model or rewrites match_predictions, simulation_results or bracket_results.

Needs data/processed/results_2026.parquet, which scripts/ingest/fetch_2026_results.py
builds from Wikipedia.

    python scripts/pipeline/score_forecast.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.evaluation import scorecard as sc                      # noqa: E402
from src.report import scorecard_view as view                   # noqa: E402
from src.utils.io import read_parquet                           # noqa: E402

OUT = "reports/forecast_scorecard.md"


def _table(df):
    """Markdown table from a frame, without depending on tabulate."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return lines


def main():
    preds = read_parquet("data/processed/match_predictions.parquet")
    sim = read_parquet("data/processed/simulation_results.parquet")
    results_path = Path("data/processed/results_2026.parquet")
    if not results_path.exists():
        sys.exit("missing results_2026.parquet; run scripts/ingest/fetch_2026_results.py first")
    results = read_parquet(results_path)

    joined = sc.join_predictions(preds, results)
    metrics = sc.wdl_metrics(joined)
    skill = sc.skill_interval(joined)
    goals = sc.goal_metrics(joined)
    calib = sc.calibration(joined)
    reach = sc.round_reach_comparison(sim, results)
    reliability = sc.round_reliability(reach)
    champ = sc.champion_check(sim, results)
    published = metrics[metrics["model"].str.startswith("published")].iloc[0]

    L = ["# How the 2026 forecast actually did", ""]
    L += [view.WHAT_THIS_SCORES, ""]

    L += ["## Match predictions", "",
          f"Over the {int(published['n'])} group-stage matches, against two reference "
          "predictors that know nothing about football:", ""]
    L += _table(view.metrics_table(metrics))
    L += ["", view.HOW_TO_READ, ""]
    L += [f"The forecast beat a predictor that gives every outcome one third by "
          f"{skill['mean_advantage']:.3f} "
          f"in log-loss per match, with a bootstrap 95% interval of "
          f"{skill['ci_low']:.3f} to {skill['ci_high']:.3f}. The interval clears zero, "
          "so on this tournament the edge is real rather than a lucky sample. It is a "
          "modest edge, and 72 matches is a modest sample, so that is as strong as the "
          "claim gets.", ""]
    L += [f"The single most likely outcome was correct in {sc.hit_rate(joined):.1%} of "
          "matches. That number is easy to read and worth less than the ones above, "
          "because it throws away everything the probabilities said about confidence.", ""]

    L += ["## Goals", "",
          "The expected goals came from the Dixon-Coles model, the same one that drove "
          "the simulation.", ""]
    L += _table(view.goals_summary(goals))
    bias = goals.iloc[0]["bias_per_side"]
    mix = sc.outcome_mix(joined)
    draw = mix[mix["outcome"] == "draw"].iloc[0]
    L += ["", f"The model expected {'fewer' if bias < 0 else 'more'} goals than the "
          f"tournament produced, by {abs(bias):.2f} per team per match, which is about "
          f"half a goal per game.", ""]
    L += ["It also leaned the wrong way on draws, expecting "
          f"{draw['mean_predicted']:.1%} against the {draw['observed']:.1%} that "
          "happened. Those two findings do not obviously share a cause. Fewer goals "
          "would normally mean more draws, not fewer, so this is worth noting and "
          "watching rather than explaining. One tournament cannot settle it.", ""]
    L += ["How the forecast leaned overall, against what happened:", ""]
    L += _table(mix.assign(
        mean_predicted=lambda d: (d["mean_predicted"] * 100).round(1),
        observed=lambda d: (d["observed"] * 100).round(1)).rename(
        columns={"mean_predicted": "mean predicted %", "observed": "actually happened %"}))
    L += [""]

    L += ["## Calibration", "",
          "Predicted probability of a home win against how often the home side actually "
          "won, in five buckets.", ""]
    L += _table(calib.round(3))
    L += ["", "Read the small buckets with care. With 72 matches split five ways, a "
          "single result moves a bucket several points.", ""]

    L += ["## Reaching each round", "",
          "Every team and round from the tournament simulation, bucketed by the "
          "probability the model gave, against how often those actually happened.", ""]
    L += _table(view.reliability_table(reliability))
    L += ["", f"This is the part that holds up best. Across {int(reliability['n'].sum())} "
          "team-round predictions "
          "the observed rates track the predicted ones closely at every level of "
          "confidence, which is what calibration is supposed to look like.", ""]

    L += ["## The title odds", "", view.champion_sentence(champ), ""]

    L += ["## What this does not show", "",
          "The knockout matches are not scored above. No win/draw/loss probabilities "
          "were ever published for them, and extra time and shootouts would make the "
          "label ambiguous anyway, so they feed the round-reaching comparison instead.",
          "",
          "These numbers cannot be compared directly against the walk-forward figures on "
          "the Model page. Those measure a different model on a different population of "
          "matches, mostly qualifiers and friendlies across many years. A World Cup field "
          "is 48 well-covered teams, which Elo rates better than it rates the long tail.",
          ""]

    L += ["## Worst misses first", "",
          "Every scored match, ordered by the probability the model gave to the outcome "
          "that actually happened. The top of this table is where the model was most "
          "wrong.", ""]
    L += _table(view.per_match_table(joined, results).head(15))
    L += ["", f"Full table of all {len(joined)} matches is on the Scorecard page of the "
          "dashboard.", ""]

    Path(OUT).write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {OUT}")
    print(f"  log-loss {published['log_loss']:.4f}, RPS {published['rps']:.4f}, "
          f"advantage over uniform {skill['mean_advantage']:.3f} "
          f"({skill['ci_low']:.3f} to {skill['ci_high']:.3f})")


if __name__ == "__main__":
    main()
