from src.ingest.apifootball import build_request

def test_build_request_uses_verified_base_and_auth_header():
    url, headers, params = build_request("fixtures", {"league": 1, "season": 2026}, api_key="KEY")
    assert url == "https://v3.football.api-sports.io/fixtures"
    assert headers["x-apisports-key"] == "KEY"
    assert params == {"league": 1, "season": 2026}

import pandas as pd
from src.schema import CANON_FIXTURE_COLS
from src.ingest.apifootball import normalize_fixtures

def test_normalize_fixtures_parses_nested_payload():
    payload = {"response": [{
        "fixture": {"id": 101, "date": "2026-06-11T19:00:00+00:00",
                    "status": {"short": "NS"}, "venue": {"name": "MetLife"}},
        "league": {"name": "World Cup", "season": 2026, "round": "Group Stage - 1"},
        "teams": {"home": {"name": "Mexico"}, "away": {"name": "Poland"}},
    }]}
    out = normalize_fixtures(payload)
    assert list(out.columns) == CANON_FIXTURE_COLS
    assert out.loc[0, "fixture_id"] == 101
    assert out.loc[0, "status"] == "NS"
    assert str(out.loc[0, "date"]) == "2026-06-11 19:00:00"   # tz stripped

from src.schema import CANON_LINEUP_COLS, CANON_INJURY_COLS
from src.ingest.apifootball import normalize_lineups, normalize_injuries

def test_normalize_lineups_flags_starters_and_subs():
    payload = {"response": [{
        "team": {"name": "Mexico"}, "formation": "4-3-3",
        "startXI": [{"player": {"name": "Ochoa", "pos": "G"}}],
        "substitutes": [{"player": {"name": "Malagon", "pos": "G"}}],
    }]}
    out = normalize_lineups(payload, fixture_id=101)
    assert list(out.columns) == CANON_LINEUP_COLS
    assert out.loc[out["player"] == "Ochoa", "is_starter"].iloc[0] == True
    assert out.loc[out["player"] == "Malagon", "is_starter"].iloc[0] == False
    assert (out["formation"] == "4-3-3").all()

def test_normalize_injuries_parses_payload():
    payload = {"response": [{
        "player": {"name": "Gavi", "type": "Missing Fixture", "reason": "Knee Injury"},
        "team": {"name": "Spain"},
        "fixture": {"date": "2026-06-10T12:00:00+00:00"},
    }]}
    out = normalize_injuries(payload)
    assert list(out.columns) == CANON_INJURY_COLS
    assert out.loc[0, "player"] == "Gavi"
    assert out.loc[0, "reason"] == "Knee Injury"
    assert out.loc[0, "status"] == "Missing Fixture"


def test_normalize_results_finished_only_with_aliases():
    from src.ingest.apifootball import normalize_results
    payload = {"response": [
        {"fixture": {"id": 1, "date": "2026-06-12T02:00:00+00:00", "status": {"short": "FT"},
                     "venue": {"name": "X"}},
         "league": {"name": "World Cup", "season": 2026, "round": "Group D"},
         "teams": {"home": {"name": "USA"}, "away": {"name": "Paraguay"}},
         "goals": {"home": 2, "away": 1}},
        {"fixture": {"id": 2, "date": "2026-06-13T00:00:00+00:00", "status": {"short": "NS"},
                     "venue": {"name": "Y"}},
         "league": {"name": "World Cup", "season": 2026, "round": "Group A"},
         "teams": {"home": {"name": "Korea Republic"}, "away": {"name": "Czech Republic"}},
         "goals": {"home": None, "away": None}},
    ]}
    out = normalize_results(payload)
    assert len(out) == 1                                     # NS (not started) excluded
    r = out.iloc[0]
    assert r["home_team"] == "United States"                 # alias applied
    assert r["away_team"] == "Paraguay"
    assert (r["home_score"], r["away_score"]) == (2, 1)
