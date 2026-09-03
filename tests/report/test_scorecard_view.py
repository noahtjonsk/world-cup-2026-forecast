import pandas as pd

from src.report.scorecard_view import (
    HOW_TO_READ, WHAT_THIS_SCORES, champion_sentence, goals_summary, headline,
    metrics_table, per_match_table, reliability_table,
)


def _metrics():
    return pd.DataFrame([
        {"model": "published (Elo baseline)", "n": 72, "log_loss": 0.900955,
         "rps": 0.164310, "brier": 0.533447},
        {"model": "uniform 1/3", "n": 72, "log_loss": 1.098612,
         "rps": 0.231481, "brier": 0.666667},
    ])


def _skill():
    return {"mean_advantage": 0.1977, "ci_low": 0.0699, "ci_high": 0.3206,
            "n": 72, "beats_uniform": True}


def _joined():
    return pd.DataFrame([
        dict(match_id="m1", date="2026-06-11", home_team="Mexico", away_team="South Africa",
             p_home=0.78, p_draw=0.15, p_away=0.07, exp_goals_home=2.7,
             exp_goals_away=0.4, actual="H"),
        dict(match_id="m2", date="2026-06-12", home_team="Turkey", away_team="Iran",
             p_home=0.40, p_draw=0.32, p_away=0.28, exp_goals_home=1.3,
             exp_goals_away=1.1, actual="A"),
    ])


def _results():
    return pd.DataFrame([
        dict(match_id="m1", home_score=2, away_score=0),
        dict(match_id="m2", home_score=0, away_score=1),
    ])


def test_headline_reports_the_published_row_not_a_reference_row():
    h = dict(headline(_metrics(), _skill()))
    assert h["Matches scored"] == "72"
    assert h["Log-loss"] == "0.901"          # the published row, not uniform's 1.099
    # Point estimate only: the full interval overflowed the metric tile, and the
    # paragraph beneath it carries the interval instead.
    assert h["Advantage over uniform"] == "0.198"


def test_metrics_table_rounds_without_reordering():
    t = metrics_table(_metrics())
    assert t["log_loss"].tolist() == [0.9010, 1.0986]
    assert t["model"].tolist() == _metrics()["model"].tolist()


def test_goals_summary_names_the_direction_of_the_bias():
    under = pd.DataFrame([{"n_matches": 72, "mae_per_side": 0.92, "bias_per_side": -0.24,
                           "mean_predicted_goals": 2.50, "mean_actual_goals": 2.99}])
    rows = goals_summary(under)["measure"].tolist()
    assert any("fewer than actual" in r for r in rows)

    over = under.copy()
    over["bias_per_side"] = 0.24
    assert any("more than actual" in r for r in goals_summary(over)["measure"].tolist())


def test_reliability_table_converts_to_percentages():
    rel = pd.DataFrame([{"bucket": "(0.25, 0.5]", "n": 31, "mean_predicted": 0.370874,
                         "observed_rate": 0.419355}])
    out = reliability_table(rel)
    assert out["mean predicted %"].iloc[0] == 37.1
    assert out["actually happened %"].iloc[0] == 41.9
    assert "cases" in out.columns


def test_per_match_table_puts_the_worst_miss_first():
    t = per_match_table(_joined(), _results())
    # m2 was an away win given only 28%, m1 a home win given 78%
    assert t.iloc[0]["home_team"] == "Turkey"
    assert t.iloc[0]["prob given to what happened (%)"] == 28.0
    assert t.iloc[0]["score"] == "0-1"
    assert t.iloc[0]["predicted"] == "home win" and t.iloc[0]["happened"] == "away win"
    assert t.iloc[-1]["home_team"] == "Mexico"


def test_champion_sentence_states_the_result_and_refuses_to_oversell_it():
    s = champion_sentence({"champion": "Spain", "runner_up": "Argentina",
                           "predicted_prob": 0.253, "predicted_rank": 1, "n_teams": 48,
                           "favorite": "Spain", "favorite_prob": 0.253})
    assert "Spain won" in s and "Argentina" in s and "25.3%" in s
    assert "not evidence of skill" in s      # the caveat is not optional
    assert champion_sentence(None).startswith("No final result")


def test_shared_copy_names_the_model_that_was_actually_published():
    assert "Elo baseline" in WHAT_THIS_SCORES
    assert "CatBoost is not scored" in WHAT_THIS_SCORES
    assert "lower-is-better" in HOW_TO_READ


def test_champion_sentence_does_not_claim_rank_one_for_a_lower_pick():
    """The winner is not always the favorite; the sentence must say which it was."""
    s = champion_sentence({"champion": "Spain", "runner_up": "Argentina",
                           "predicted_prob": 0.05, "predicted_rank": 7, "n_teams": 48,
                           "favorite": "Brazil", "favorite_prob": 0.22})
    assert "number 7 pick of 48" in s
    assert "favorite" not in s                # it was not the favorite
    assert "not evidence of skill" in s


def test_champion_sentence_handles_a_winner_missing_from_the_odds():
    s = champion_sentence({"champion": "Spain", "runner_up": "Argentina",
                           "predicted_prob": float("nan"), "predicted_rank": -1,
                           "n_teams": 48, "favorite": "Brazil", "favorite_prob": 0.22})
    assert "nan" not in s.lower()             # never render a NaN percentage
    assert "does not appear in the title odds" in s
    assert "Brazil" in s
