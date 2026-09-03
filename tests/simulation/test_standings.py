import pandas as pd
import numpy as np
from src.simulation.standings import group_table, rank_group, rank_third_places

def test_group_table_points_gd_gf():
    results = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0},
        {"home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1},
        {"home_team": "B", "away_team": "C", "home_score": 0, "away_score": 3},
    ])
    tbl = group_table(results)
    assert list(tbl.columns) == ["team", "played", "won", "drawn", "lost", "gf", "ga", "gd", "points"]
    idx = tbl.set_index("team")
    assert idx.loc["A", "points"] == 4 and idx.loc["A", "gd"] == 2 and idx.loc["A", "gf"] == 3
    assert idx.loc["C", "points"] == 4 and idx.loc["C", "gd"] == 3 and idx.loc["C", "gf"] == 4
    assert idx.loc["B", "points"] == 0 and idx.loc["B", "gd"] == -5


def test_rank_group_points_then_gd():
    results = pd.DataFrame([
        {"home_team": "A", "away_team": "B", "home_score": 2, "away_score": 0},
        {"home_team": "A", "away_team": "C", "home_score": 1, "away_score": 1},
        {"home_team": "B", "away_team": "C", "home_score": 0, "away_score": 3},
    ])
    # A & C both 4 pts; C's GD (+3) beats A's (+2); B last
    assert rank_group(results, np.random.default_rng(0)) == ["C", "A", "B"]


def test_rank_group_head_to_head_breaks_overall_tie():
    # A and B finish identical on points(4)/GD(0)/GF(3); B beat A head-to-head 2-1
    results = pd.DataFrame([
        {"home_team": "B", "away_team": "A", "home_score": 2, "away_score": 1},
        {"home_team": "A", "away_team": "C", "home_score": 2, "away_score": 1},
        {"home_team": "A", "away_team": "D", "home_score": 0, "away_score": 0},
        {"home_team": "B", "away_team": "C", "home_score": 1, "away_score": 1},
        {"home_team": "D", "away_team": "B", "home_score": 1, "away_score": 0},
        {"home_team": "C", "away_team": "D", "home_score": 1, "away_score": 1},
    ])
    assert rank_group(results, np.random.default_rng(0)) == ["D", "B", "A", "C"]


def test_rank_third_places_top_k_by_points_then_gd():
    thirds = pd.DataFrame([
        {"group": "A", "team": "TA", "points": 4, "gd": 1, "gf": 3},
        {"group": "B", "team": "TB", "points": 4, "gd": 2, "gf": 4},
        {"group": "C", "team": "TC", "points": 3, "gd": 0, "gf": 2},
        {"group": "D", "team": "TD", "points": 6, "gd": 3, "gf": 5},
    ])
    best = rank_third_places(thirds, k=2, rng=np.random.default_rng(0))
    assert best == [("D", "TD"), ("B", "TB")]   # TD 6pts; then TB (gd2) over TA (gd1)


def test_fast_rank_matches_pandas_rank_on_random_groups():
    # The vectorized tournament loop must reproduce rank_group exactly: same
    # results + same rng state => same order (including drawn-lots blocks).
    import numpy as np
    import pandas as pd
    from src.simulation.standings import rank_group, rank_group_fast
    rng_data = np.random.default_rng(0)
    for trial in range(100):
        teams = [f"T{i}" for i in range(4)]
        rows = [{"home_team": teams[i], "away_team": teams[j],
                 "home_score": int(rng_data.integers(0, 4)),
                 "away_score": int(rng_data.integers(0, 4))}
                for i in range(4) for j in range(i + 1, 4)]
        df = pd.DataFrame(rows)
        tuples = [(r["home_team"], r["away_team"], r["home_score"], r["away_score"]) for r in rows]
        slow = rank_group(df, np.random.default_rng(trial))
        fast = rank_group_fast(tuples, np.random.default_rng(trial))
        assert slow == fast, f"trial {trial}: {slow} != {fast}"


def test_fast_third_places_matches_pandas_version():
    import numpy as np
    import pandas as pd
    from src.simulation.standings import rank_third_places, rank_third_places_fast
    rows = [{"group": g, "team": f"3rd{g}", "points": int(p), "gd": int(d), "gf": int(f)}
            for g, p, d, f in [("A", 4, 1, 3), ("B", 4, 1, 3), ("C", 6, 2, 5),
                               ("D", 3, -1, 2), ("E", 4, 0, 4), ("F", 2, -2, 1)]]
    slow = rank_third_places(pd.DataFrame(rows), 4, np.random.default_rng(5))
    fast = rank_third_places_fast(
        [(r["group"], r["team"], r["points"], r["gd"], r["gf"]) for r in rows],
        4, np.random.default_rng(5))
    assert slow == fast
