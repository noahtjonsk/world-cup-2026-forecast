import pandas as pd
from src.schema import CANON_MATCH_COLS
from src.ingest.results import load_results

def test_load_results_normalizes_martj42_csv(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2024-06-15,Germany,Scotland,5,1,UEFA Euro,Munich,Germany,False\n"
        "2024-06-16,Spain,Croatia,3,0,UEFA Euro,Berlin,Germany,True\n"
    )
    out = load_results(csv)
    assert list(out.columns) == CANON_MATCH_COLS
    assert out.loc[0, "source"] == "results"
    assert out.loc[0, "neutral"] == False and out.loc[1, "neutral"] == True
    assert out["match_id"].nunique() == 2


def _wc_fixtures():
    return pd.DataFrame([
        {"fixture_id": 900007, "date": pd.Timestamp("2026-06-12"), "competition": "World Cup",
         "season": "2026", "stage": "Group D", "home_team": "United States",
         "away_team": "Paraguay", "status": "SCHEDULED", "venue": None, "neutral": True,
         "source": "wikipedia_2026"},
    ])


def test_align_results_to_fixtures_uses_our_date_and_schema():
    from src.ingest.results import align_results_to_fixtures
    from src.schema import CANON_MATCH_COLS
    from src.utils.ids import make_match_id
    # API kickoff rolled to June 13 UTC (evening US match), must still match our
    # June 12 fixture and take OUR date so match_id aligns with the rest of the repo
    res = pd.DataFrame([{"date": pd.Timestamp("2026-06-13 02:00:00"),
                         "home_team": "United States", "away_team": "Paraguay",
                         "home_score": 2, "away_score": 1, "status": "FT"}])
    matched, unmatched = align_results_to_fixtures(res, _wc_fixtures())
    assert unmatched.empty
    assert list(matched.columns) == CANON_MATCH_COLS
    m = matched.iloc[0]
    assert m["date"] == pd.Timestamp("2026-06-12")           # our fixture date wins
    assert m["match_id"] == make_match_id(pd.Timestamp("2026-06-12"), "United States", "Paraguay")
    assert m["competition"] == "World Cup" and m["neutral"] == True


def test_align_results_unmatched_reported_not_dropped():
    from src.ingest.results import align_results_to_fixtures
    res = pd.DataFrame([{"date": pd.Timestamp("2026-06-12"), "home_team": "Atlantis",
                         "away_team": "Paraguay", "home_score": 1, "away_score": 0,
                         "status": "FT"}])
    matched, unmatched = align_results_to_fixtures(res, _wc_fixtures())
    assert matched.empty and len(unmatched) == 1


def test_append_new_matches_idempotent():
    from src.ingest.results import align_results_to_fixtures, append_new_matches
    res = pd.DataFrame([{"date": pd.Timestamp("2026-06-12"),
                         "home_team": "United States", "away_team": "Paraguay",
                         "home_score": 2, "away_score": 1, "status": "FT"}])
    matched, _ = align_results_to_fixtures(res, _wc_fixtures())
    once = append_new_matches(matched.iloc[0:0], matched)    # append to empty -> 1 row
    grown = append_new_matches(once, matched)                # appending again adds nothing
    assert len(once) == len(grown) == 1


def test_mark_fixtures_finished():
    from src.ingest.results import mark_fixtures_finished
    res = pd.DataFrame([{"date": pd.Timestamp("2026-06-12"),
                         "home_team": "United States", "away_team": "Paraguay",
                         "home_score": 2, "away_score": 1, "status": "FT"}])
    fx = mark_fixtures_finished(_wc_fixtures(), res)
    assert fx.iloc[0]["status"] == "FT"
