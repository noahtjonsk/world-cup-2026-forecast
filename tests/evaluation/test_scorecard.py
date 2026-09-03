import numpy as np
import pandas as pd
import pytest

from src.evaluation.scorecard import (
    actual_rounds_reached, calibration, champion_check, goal_metrics, hit_rate,
    join_predictions, outcome_labels, round_reach_comparison, round_reliability,
    wdl_metrics,
)


def _results():
    return pd.DataFrame([
        # group stage: a home win, a draw, an away win
        dict(match_id="m1", date="2026-06-11", stage="A", home_team="Mexico",
             away_team="South Africa", home_score=2, away_score=0, decided_by="regulation"),
        dict(match_id="m2", date="2026-06-12", stage="A", home_team="Turkey",
             away_team="Iran", home_score=1, away_score=1, decided_by="regulation"),
        dict(match_id="m3", date="2026-06-13", stage="B", home_team="Japan",
             away_team="Brazil", home_score=0, away_score=3, decided_by="regulation"),
        # knockouts
        dict(match_id="k1", date="2026-06-28", stage="Round of 32", home_team="Spain",
             away_team="Austria", home_score=3, away_score=0, decided_by="regulation"),
        dict(match_id="k2", date="2026-07-19", stage="Final", home_team="Spain",
             away_team="Argentina", home_score=1, away_score=0, decided_by="aet"),
    ])


def _predictions():
    return pd.DataFrame([
        dict(match_id="m1", date="2026-06-11", home_team="Mexico", away_team="South Africa",
             snapshot_date="2026-06-11", p_home=0.78, p_draw=0.15, p_away=0.07,
             exp_goals_home=2.7, exp_goals_away=0.4, source="live"),
        dict(match_id="m2", date="2026-06-12", home_team="Turkey", away_team="Iran",
             snapshot_date="2026-06-11", p_home=0.40, p_draw=0.32, p_away=0.28,
             exp_goals_home=1.3, exp_goals_away=1.1, source="live"),
        dict(match_id="m3", date="2026-06-13", home_team="Japan", away_team="Brazil",
             snapshot_date="2026-06-11", p_home=0.20, p_draw=0.25, p_away=0.55,
             exp_goals_home=0.9, exp_goals_away=1.8, source="live"),
    ])


def test_outcome_labels_cover_all_three_results():
    assert list(outcome_labels(_results())) == ["H", "D", "A", "H", "H"]


def test_join_refuses_to_score_a_subset():
    preds = _predictions()
    partial = _results().iloc[1:]              # m1 has no result any more
    with pytest.raises(ValueError, match="no result"):
        join_predictions(preds, partial)


def test_join_attaches_the_actual_outcome():
    joined = join_predictions(_predictions(), _results())
    assert len(joined) == 3
    assert list(joined["actual"]) == ["H", "D", "A"]


def test_published_beats_the_reference_predictors_on_this_fixture():
    joined = join_predictions(_predictions(), _results())
    m = wdl_metrics(joined).set_index("model")
    assert set(m.index) == {"published (Elo baseline)", "uniform 1/3", "always home"}
    # the fixture predictions point the right way in all three matches
    assert m.loc["published (Elo baseline)", "log_loss"] < m.loc["uniform 1/3", "log_loss"]
    assert m.loc["published (Elo baseline)", "log_loss"] < m.loc["always home", "log_loss"]
    assert (m["n"] == 3).all()


def test_uniform_reference_has_the_analytic_log_loss():
    joined = join_predictions(_predictions(), _results())
    m = wdl_metrics(joined).set_index("model")
    assert m.loc["uniform 1/3", "log_loss"] == pytest.approx(np.log(3), abs=1e-9)


def test_goal_metrics_report_bias_with_a_readable_sign():
    joined = join_predictions(_predictions(), _results())
    g = goal_metrics(joined).iloc[0]
    assert g["n_matches"] == 3
    # predicted 3.1 + 2.4 + 2.7 = 8.2 goals against 2 + 2 + 3 = 7 actual
    assert g["mean_predicted_goals"] == pytest.approx(8.2 / 3)
    assert g["mean_actual_goals"] == pytest.approx(7 / 3)
    assert g["bias_per_side"] > 0            # over-predicted, so positive
    assert g["mae_per_side"] >= 0


def test_hit_rate_counts_the_modal_pick():
    joined = join_predictions(_predictions(), _results())
    # m1 picks H (right), m2 picks H but was D (wrong), m3 picks A (right)
    assert hit_rate(joined) == pytest.approx(2 / 3)


def test_calibration_returns_buckets_without_crashing_on_small_n():
    c = calibration(join_predictions(_predictions(), _results()), n_bins=2)
    assert len(c) >= 1


def test_deepest_round_reached_ignores_the_group_and_third_place():
    reached = actual_rounds_reached(_results())
    assert reached["Spain"] == "F"           # played the final
    assert reached["Argentina"] == "F"       # losing finalist still reached it
    assert reached["Austria"] == "R32"
    assert "Mexico" not in reached           # group stage only


def test_round_reach_comparison_marks_deeper_runs_as_reaching_earlier_rounds():
    sim = pd.DataFrame([
        dict(tournament="2026", team="Spain", round="R32", prob=0.9, ci_low=0, ci_high=1),
        dict(tournament="2026", team="Spain", round="F", prob=0.4, ci_low=0, ci_high=1),
        dict(tournament="2026", team="Mexico", round="R32", prob=0.7, ci_low=0, ci_high=1),
        dict(tournament="2026", team="Spain", round="W", prob=0.25, ci_low=0, ci_high=1),
    ])
    cmp = round_reach_comparison(sim, _results()).set_index(["team", "round"])
    assert cmp.loc[("Spain", "R32"), "actual"] == 1     # reaching F implies reaching R32
    assert cmp.loc[("Spain", "F"), "actual"] == 1
    assert cmp.loc[("Mexico", "R32"), "actual"] == 0    # never left the group
    assert ("Spain", "W") not in cmp.index              # W is not a reachable round here


def test_round_reliability_buckets_predictions_against_outcomes():
    reach = pd.DataFrame({
        "team": list("abcdef"), "round": ["R32"] * 6,
        "predicted": [0.05, 0.05, 0.8, 0.9, 0.85, 0.05],
        "actual": [0, 0, 1, 1, 0, 0],
    })
    rel = round_reliability(reach)
    assert rel["n"].sum() == 6
    low = rel[rel["mean_predicted"] < 0.2].iloc[0]
    assert low["observed_rate"] == 0.0


def test_champion_check_reports_the_probability_and_the_rank():
    sim = pd.DataFrame([
        dict(tournament="2026", team="Spain", round="W", prob=0.253, ci_low=0, ci_high=1),
        dict(tournament="2026", team="Argentina", round="W", prob=0.162, ci_low=0, ci_high=1),
        dict(tournament="2026", team="France", round="W", prob=0.129, ci_low=0, ci_high=1),
    ])
    c = champion_check(sim, _results())
    assert c["champion"] == "Spain" and c["runner_up"] == "Argentina"
    assert c["predicted_prob"] == pytest.approx(0.253)
    assert c["predicted_rank"] == 1
    assert c["favorite"] == "Spain"
    assert c["n_teams"] == 3


def test_champion_check_handles_a_winner_the_model_ranked_low():
    sim = pd.DataFrame([
        dict(tournament="2026", team="Brazil", round="W", prob=0.5, ci_low=0, ci_high=1),
        dict(tournament="2026", team="Spain", round="W", prob=0.01, ci_low=0, ci_high=1),
    ])
    c = champion_check(sim, _results())
    assert c["champion"] == "Spain"
    assert c["predicted_rank"] == 2
    assert c["favorite"] == "Brazil"


def test_skill_interval_detects_a_forecast_that_beats_uniform():
    from src.evaluation.scorecard import skill_interval
    joined = join_predictions(_predictions(), _results())
    s = skill_interval(joined, n_boot=2000, seed=1)
    assert s["n"] == 3
    assert s["mean_advantage"] > 0                   # fixture predictions are informative
    assert s["ci_low"] <= s["mean_advantage"] <= s["ci_high"]


def test_skill_interval_does_not_claim_skill_for_a_uniform_forecast():
    # A forecast that says 1/3 to everything has, by construction, no advantage at all.
    preds = _predictions()
    preds[["p_home", "p_draw", "p_away"]] = 1 / 3
    from src.evaluation.scorecard import skill_interval
    s = skill_interval(join_predictions(preds, _results()), n_boot=2000, seed=1)
    assert s["mean_advantage"] == pytest.approx(0.0, abs=1e-12)
    assert s["beats_uniform"] is False


def test_outcome_mix_compares_average_prediction_to_observed_frequency():
    from src.evaluation.scorecard import outcome_mix
    mix = outcome_mix(join_predictions(_predictions(), _results())).set_index("outcome")
    assert list(mix.index) == ["home win", "draw", "away win"]
    # fixture: one H, one D, one A, so each observed frequency is a third
    assert mix.loc["draw", "observed"] == pytest.approx(1 / 3)
    assert mix.loc["home win", "mean_predicted"] == pytest.approx((0.78 + 0.40 + 0.20) / 3)
    assert mix["observed"].sum() == pytest.approx(1.0)
