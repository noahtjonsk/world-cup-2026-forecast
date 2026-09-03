import pandas as pd
from src.report.bracket_view import slot_candidates, modal_pairings, OFFICIAL_MATCH_NUMBERS


def _bracket():
    # one QF slot played 10 times: A-C 6x (A wins 4), A-D 2x (A wins 1), B-C 2x (B wins 2)
    return pd.DataFrame([
        {"tournament": "T", "round": "QF", "match_idx": 0,
         "home_team": "A", "away_team": "C", "n": 6, "home_wins": 4},
        {"tournament": "T", "round": "QF", "match_idx": 0,
         "home_team": "A", "away_team": "D", "n": 2, "home_wins": 1},
        {"tournament": "T", "round": "QF", "match_idx": 0,
         "home_team": "B", "away_team": "C", "n": 2, "home_wins": 2},
    ])


def test_slot_candidates_marginal_probabilities():
    out = slot_candidates(_bracket())
    assert list(out.columns) == ["round", "match_idx", "side", "team", "prob"]
    qf = out.set_index(["side", "team"])["prob"]
    assert abs(qf.loc[("home", "A")] - 0.8) < 1e-9      # 8 of 10 runs
    assert abs(qf.loc[("home", "B")] - 0.2) < 1e-9
    assert abs(qf.loc[("away", "C")] - 0.8) < 1e-9      # 6 + 2
    assert abs(qf.loc[("away", "D")] - 0.2) < 1e-9
    # sorted: per (round, match_idx, side) descending prob
    home = out[out["side"] == "home"]["prob"].tolist()
    assert home == sorted(home, reverse=True)


def test_modal_pairings_top_pairing_and_h2h():
    out = modal_pairings(_bracket())
    assert list(out.columns) == ["round", "match_idx", "home_team", "away_team",
                                 "pair_prob", "p_home_win", "favorite", "p_favorite"]
    r = out.iloc[0]
    assert (r["home_team"], r["away_team"]) == ("A", "C")   # most common pairing
    assert abs(r["pair_prob"] - 0.6) < 1e-9                 # 6 of 10
    assert abs(r["p_home_win"] - 4 / 6) < 1e-9              # conditional head-to-head
    assert r["favorite"] == "A" and abs(r["p_favorite"] - 4 / 6) < 1e-9


def test_modal_pairings_favorite_is_away_when_home_is_underdog():
    # nominal home side loses the tie more often than not -> the AWAY team is the
    # favorite and p_favorite is its (>= 0.5) win probability
    b = pd.DataFrame([{"tournament": "T", "round": "R16", "match_idx": 7,
                       "home_team": "Switzerland", "away_team": "Portugal",
                       "n": 100, "home_wins": 34}])
    r = modal_pairings(b).iloc[0]
    assert r["favorite"] == "Portugal"
    assert abs(r["p_favorite"] - 0.66) < 1e-9


def test_official_match_numbers_cover_the_tree():
    lens = {r: len(v) for r, v in OFFICIAL_MATCH_NUMBERS.items()}
    assert lens == {"R32": 16, "R16": 8, "QF": 4, "SF": 2, "F": 1}
    flat = [n for v in OFFICIAL_MATCH_NUMBERS.values() for n in v]
    assert len(flat) == len(set(flat)) == 31                # matches 73..102 + 104, all unique
    assert set(OFFICIAL_MATCH_NUMBERS["QF"]) == {97, 98, 99, 100}
