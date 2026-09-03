from src.schema import CANON_FEATURE_COLS

def test_feature_schema_present_and_ordered():
    assert CANON_FEATURE_COLS[:5] == ["match_id", "date", "home_team", "away_team", "snapshot_date"]
    assert CANON_FEATURE_COLS[-1] == "result"
    expected = {
        "elo_home", "elo_away", "elo_diff",
        "form_home", "form_away", "form_diff",
        "xi_quality_home", "xi_quality_away", "xi_quality_diff",
        "bench_dropoff_home", "bench_dropoff_away",
        "role_coverage_home", "role_coverage_away",
        "style_mismatch", "xt_diff",
        "rest_home", "rest_away", "rest_diff",
        "stage_code", "neutral", "host_home", "host_away",
    }
    assert expected <= set(CANON_FEATURE_COLS)
    assert len(CANON_FEATURE_COLS) == len(set(CANON_FEATURE_COLS))  # no dupes


import pandas as pd
import pytest
from src.features.snapshot import match_result, build_features

def test_match_result_labels():
    assert match_result(2, 0) == "H"
    assert match_result(1, 1) == "D"
    assert match_result(0, 3) == "A"
    assert match_result(None, 1) is None

def _fixtures():
    matches = pd.DataFrame({
        "match_id": ["M1", "M0"],
        "date": pd.to_datetime(["2026-06-10", "2026-05-01"]),
        "competition": ["WC", "Friendly"],
        "season": ["2026", "2026"],
        "home_team": ["Spain", "Spain"],
        "away_team": ["Italy", "Brazil"],
        "home_score": [2.0, 1.0],
        "away_score": [0.0, 1.0],
        "stage": ["Group", "Friendly"],
        "neutral": [True, False],
        "source": ["sb", "sb"],
    })
    ratings = pd.DataFrame({
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2027-01-01"]),
        "team": ["Spain", "Italy", "Spain"],
        "elo":  [2050.0, 1900.0, 9999.0],     # future row -> ignored
        "source": ["eloratings"] * 3,
    })
    player_stats = pd.DataFrame({
        "player": ["P1", "P2", "P3", "P4"],
        "team": ["Spain", "Spain", "Italy", "Italy"],
        "season": ["2024-2025"] * 4,
        "position": ["ST", "CB", "ST", "CB"],
        "metric": ["vaep_p90"] * 4,
        "value": [0.9, 0.5, 0.4, 0.2],
        "source": ["vaep"] * 4,
    })
    team_style = pd.DataFrame({
        "team": ["Spain", "Italy"],
        "season": ["2026", "2026"],
        "metric": ["mean_xt", "mean_xt"],
        "value": [0.3, 0.1],
        "source": ["vaep"] * 2,
    })
    lineups = pd.DataFrame({
        "match_id": ["M1"] * 4,
        "team": ["Spain", "Spain", "Italy", "Italy"],
        "player": ["P1", "P2", "P3", "P4"],
        "is_starter": [True, True, True, True],
        "position": ["ST", "CB", "ST", "CB"],
    })
    return matches, ratings, player_stats, team_style, lineups

def test_build_features_row_shape_and_values():
    from src.schema import CANON_FEATURE_COLS
    matches, ratings, player_stats, team_style, lineups = _fixtures()
    out = build_features("M1", matches, ratings, player_stats, team_style, lineups,
                         months=24, host_teams=())
    assert list(out.columns) == CANON_FEATURE_COLS
    r = out.iloc[0]
    assert r["elo_home"] == 2050.0 and r["elo_away"] == 1900.0 and r["elo_diff"] == 150.0
    assert r["result"] == "H"
    assert r["neutral"] == 1
    assert abs(r["xt_diff"] - 0.2) < 1e-9
    assert r["xi_quality_home"] == r["xi_quality_home"]   # not NaN (lineups + player stats present)


def test_build_features_no_lineups_squad_nan():
    # Future fixture path: no lineup rows -> squad features NaN, rest of row still valid.
    matches, ratings, player_stats, team_style, _ = _fixtures()
    empty_lineups = pd.DataFrame(columns=["match_id", "team", "player", "is_starter"])
    out = build_features("M1", matches, ratings, player_stats, team_style, empty_lineups, months=24)
    assert list(out.columns) == CANON_FEATURE_COLS
    r = out.iloc[0]
    assert pd.isna(r["xi_quality_home"]) and pd.isna(r["xi_quality_away"]) and pd.isna(r["xi_quality_diff"])
    assert r["elo_home"] == 2050.0          # non-squad features still populated
    assert r["result"] == "H"


def test_build_features_unknown_match_id_raises():
    matches, ratings, player_stats, team_style, lineups = _fixtures()
    with pytest.raises(KeyError):
        build_features("NOPE", matches, ratings, player_stats, team_style, lineups, months=24)


def test_build_features_lineups_without_match_id_raises():
    # A non-empty lineups frame keyed by fixture_id (canonical) instead of match_id is a
    # wiring bug -> raise loudly rather than silently degrade squad features to NaN.
    matches, ratings, player_stats, team_style, _ = _fixtures()
    bad = pd.DataFrame({
        "fixture_id": ["F1", "F1"], "team": ["Spain", "Italy"],
        "player": ["P1", "P3"], "is_starter": [True, True],
    })
    with pytest.raises(KeyError):
        build_features("M1", matches, ratings, player_stats, team_style, bad, months=24)


def test_build_features_wrong_season_style_is_nan():
    # team_style is season-scoped: a style table for a different season yields NaN style
    # features (the season-as-of guard is the caller's responsibility, surfaced here).
    matches, ratings, player_stats, _, lineups = _fixtures()
    other_season_style = pd.DataFrame({
        "team": ["Spain", "Italy"], "season": ["2099", "2099"],
        "metric": ["mean_xt", "mean_xt"], "value": [0.3, 0.1], "source": ["vaep"] * 2,
    })
    out = build_features("M1", matches, ratings, player_stats, other_season_style, lineups, months=24)
    r = out.iloc[0]
    assert pd.isna(r["style_mismatch"]) and pd.isna(r["xt_diff"])


def test_build_features_ignores_post_kickoff_data():
    matches, ratings, player_stats, team_style, lineups = _fixtures()
    base = build_features("M1", matches, ratings, player_stats, team_style, lineups, months=24)

    # Poison every dynamic input with absurd FUTURE rows (after the 2026-06-10 kickoff).
    poisoned_ratings = pd.concat([ratings, pd.DataFrame({
        "date": pd.to_datetime(["2026-08-01", "2026-08-01"]),
        "team": ["Spain", "Italy"], "elo": [5000.0, 5000.0], "source": ["eloratings"] * 2,
    })], ignore_index=True)
    poisoned_matches = pd.concat([matches, pd.DataFrame({
        "match_id": ["MF"], "date": pd.to_datetime(["2026-07-01"]),
        "competition": ["WC"], "season": ["2026"], "home_team": ["Spain"], "away_team": ["Italy"],
        "home_score": [9.0], "away_score": [0.0], "stage": ["Group"], "neutral": [True], "source": ["sb"],
    })], ignore_index=True)
    poisoned_players = pd.concat([player_stats, pd.DataFrame({
        "player": ["P1"], "team": ["Spain"], "season": ["2030-2031"], "position": ["ST"],
        "metric": ["vaep_p90"], "value": [999.0], "source": ["vaep"],
    })], ignore_index=True)

    after = build_features("M1", poisoned_matches, poisoned_ratings, poisoned_players,
                           team_style, lineups, months=24)

    numeric = [c for c in base.columns if c not in ("match_id", "home_team", "away_team", "result")]
    pd.testing.assert_frame_equal(base[numeric], after[numeric])


from src.features.snapshot import build_feature_table
from src.ingest.run import persist_tables
from src.utils.io import read_parquet

def test_build_feature_table_and_persist(tmp_path):
    matches, ratings, player_stats, team_style, lineups = _fixtures()
    table = build_feature_table(["M1"], matches, ratings, player_stats, team_style, lineups, months=24)
    assert list(table.columns) == CANON_FEATURE_COLS
    assert len(table) == 1

    paths = persist_tables({"matchup_features": table}, out_dir=tmp_path / "processed")
    assert paths["matchup_features"].exists()
    back = read_parquet(paths["matchup_features"])
    assert list(back.columns) == CANON_FEATURE_COLS
    assert back.loc[0, "result"] == "H"


def test_build_features_matches_players_across_name_spellings():
    # Lineups carry Wikipedia spellings, player_stats carry FBref spellings, the
    # squad join must normalize both sides (accents/case), not require raw equality.
    matches, ratings, player_stats, team_style, lineups = _fixtures()
    player_stats = player_stats.assign(
        player=["Kylián Mbappé", "Rúben Días", "Çalhanoğlu", "João Félix"])
    lineups = lineups.assign(
        player=["Kylian Mbappe", "Ruben Dias", "Calhanoglu", "Joao Felix"])
    out = build_features("M1", matches, ratings, player_stats, team_style, lineups,
                         months=24, host_teams=())
    r = out.iloc[0]
    assert r["xi_quality_home"] == r["xi_quality_home"]      # not NaN: all 4 matched
    # Spain starters P-quality mean must reflect BOTH matched players (0.9, 0.5 ranks)
    assert r["xi_quality_home"] > 0
