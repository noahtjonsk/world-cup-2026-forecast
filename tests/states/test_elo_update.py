from src.states.elo_update import _gd_multiplier, elo_update_one

def test_gd_multiplier_eloratings_bands():
    assert _gd_multiplier(0) == 1.0 and _gd_multiplier(1) == 1.0 and _gd_multiplier(-1) == 1.0
    assert _gd_multiplier(2) == 1.5 and _gd_multiplier(-2) == 1.5
    assert _gd_multiplier(3) == (11 + 3) / 8.0          # 1.75
    assert _gd_multiplier(-4) == (11 + 4) / 8.0         # 1.875

def test_elo_update_even_match_neutral():
    # 1500 v 1500 neutral, K=60, We=0.5
    assert elo_update_one(1500.0, 1500.0, 1, 0, k=60.0, neutral=True) == (1530.0, 1470.0)   # win 1-0
    assert elo_update_one(1500.0, 1500.0, 3, 0, k=60.0, neutral=True) == (1552.5, 1447.5)   # win 3-0, G=1.75
    h, a = elo_update_one(1500.0, 1500.0, 1, 1, k=60.0, neutral=True)                       # draw
    assert (h, a) == (1500.0, 1500.0)

def test_favourite_winning_gains_less_than_even_match():
    nh, _ = elo_update_one(1700.0, 1500.0, 1, 0, k=60.0, neutral=True)   # strong favourite wins
    assert 0.0 < nh - 1700.0 < 30.0                                     # gains < the even-match +30

import pandas as pd
from src.states.elo_update import seed_from_ratings, recompute_elo
from src.schema import CANON_RATING_COLS

def test_seed_from_ratings_latest_before_cutoff():
    ratings = pd.DataFrame([
        {"date": "2026-01-01", "team": "A", "elo": 1500.0, "source": "eloratings"},
        {"date": "2026-05-01", "team": "A", "elo": 1550.0, "source": "eloratings"},
        {"date": "2026-07-01", "team": "A", "elo": 9999.0, "source": "eloratings"},  # after cutoff
        {"date": "2026-03-01", "team": "B", "elo": 1480.0, "source": "eloratings"},
    ])
    assert seed_from_ratings(ratings, cutoff="2026-06-11") == {"A": 1550.0, "B": 1480.0}

def test_recompute_elo_idempotent_zero_sum_and_dated():
    matches = pd.DataFrame([
        {"date": "2026-06-12", "competition": "World Cup", "home_team": "A",
         "away_team": "B", "home_score": 1, "away_score": 0, "neutral": True},
        {"date": "2026-06-16", "competition": "World Cup", "home_team": "A",
         "away_team": "C", "home_score": 1, "away_score": 1, "neutral": True},
    ])
    seed = {"A": 1500.0, "B": 1500.0, "C": 1500.0}
    out1 = recompute_elo(matches, seed, default_k=60.0)
    out2 = recompute_elo(matches, seed, default_k=60.0)
    pd.testing.assert_frame_equal(out1, out2)                       # idempotent
    assert list(out1.columns) == CANON_RATING_COLS
    # match 1 (A beats B 1-0, neutral, K=60): A=1530, B=1470 (zero-sum vs 1500 seed)
    m1 = out1[out1["date"] == pd.Timestamp("2026-06-12")].set_index("team")["elo"]
    assert abs(m1["A"] - 1530.0) < 1e-9 and abs(m1["B"] - 1470.0) < 1e-9

def test_recompute_elo_since_ignores_pre_cutoff_matches():
    # The seed is the rating AS OF the cutoff: replaying matches from BEFORE it
    # double-counts history on top of current ratings (the bug behind the
    # corrupted Match-page elo_diff). `since` filters the replay to the cutoff.
    matches = pd.DataFrame([
        {"date": "2024-03-01", "competition": "Friendly", "home_team": "A",
         "away_team": "B", "home_score": 5, "away_score": 0, "neutral": True},   # history: already in the seed
        {"date": "2026-06-12", "competition": "World Cup", "home_team": "A",
         "away_team": "B", "home_score": 1, "away_score": 0, "neutral": True},   # new result: replay this
    ])
    seed = {"A": 1600.0, "B": 1500.0}
    out = recompute_elo(matches, seed, default_k=60.0, since="2026-06-11")
    assert (out["date"] == pd.Timestamp("2026-06-12")).all()         # pre-cutoff row excluded
    final = out.set_index("team")["elo"]
    # one update from the seeds: We(A)=1/(1+10^(-100/400))~0.640; delta=60*(1-0.640)
    assert abs(final["A"] - (1600.0 + 60.0 * (1 - 1 / (1 + 10 ** (-100 / 400))))) < 1e-6

def test_recompute_elo_since_with_no_new_matches_is_empty():
    matches = pd.DataFrame([
        {"date": "2024-03-01", "competition": "Friendly", "home_team": "A",
         "away_team": "B", "home_score": 2, "away_score": 1, "neutral": True},
    ])
    out = recompute_elo(matches, {"A": 1600.0, "B": 1500.0}, since="2026-06-11")
    assert out.empty and list(out.columns) == CANON_RATING_COLS
