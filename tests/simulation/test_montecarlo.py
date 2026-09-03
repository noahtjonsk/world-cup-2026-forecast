import numpy as np
from src.simulation.montecarlo import run_simulation, champion_odds
from src.schema import CANON_SIM_RESULT_COLS


def _setup():
    fmt = {"qualifiers_per_group": 1, "third_place_slots": 0, "rounds": ["F"]}
    groups = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    params = {"attack": {"A1": 1.0, "A2": -1.0, "B1": 1.0, "B2": -1.0},
              "defence": {k: 0.0 for k in ["A1", "A2", "B1", "B2"]},
              "home_adv": 0.0, "rho": 0.0}
    seeding = [["1A", "1B"]]
    return fmt, groups, params, seeding


def test_run_simulation_shape_intervals_and_strength_ordering():
    fmt, groups, params, seeding = _setup()
    out = run_simulation(fmt, groups, params, seeding, n_runs=300,
                         rng=np.random.default_rng(0), n_boot=50, tournament="T")
    assert list(out.columns) == CANON_SIM_RESULT_COLS
    assert set(out["round"]) == {"F", "W"}                       # stages 1..2 reported
    assert ((out["prob"] >= 0) & (out["prob"] <= 1)).all()
    assert (out["ci_low"] <= out["prob"] + 1e-9).all() and (out["prob"] <= out["ci_high"] + 1e-9).all()
    champ = out[out["round"] == "W"].set_index("team")["prob"]
    assert champ["A1"] > champ["A2"] and champ["B1"] > champ["B2"]


def test_champion_odds_sorted_descending():
    fmt, groups, params, seeding = _setup()
    out = run_simulation(fmt, groups, params, seeding, n_runs=300,
                         rng=np.random.default_rng(0), n_boot=50, tournament="T")
    odds = champion_odds(out, top=2)
    assert len(odds) == 2
    assert list(odds["prob"]) == sorted(odds["prob"], reverse=True)


def test_run_simulation_collect_bracket_counts():
    import numpy as np
    from src.schema import CANON_BRACKET_COLS
    from src.simulation.montecarlo import run_simulation
    fmt = {"qualifiers_per_group": 1, "third_place_slots": 0, "rounds": ["F"]}
    groups = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    params = {"attack": {t: 0.0 for t in ["A1", "A2", "B1", "B2"]},
              "defence": {t: 0.0 for t in ["A1", "A2", "B1", "B2"]},
              "home_adv": 0.0, "rho": 0.0}
    res, bracket = run_simulation(fmt, groups, params, [["1A", "1B"]], n_runs=50,
                                  rng=np.random.default_rng(1), tournament="T",
                                  collect_bracket=True)
    assert list(bracket.columns) == CANON_BRACKET_COLS
    g = bracket[(bracket["round"] == "F") & (bracket["match_idx"] == 0)]
    assert int(g["n"].sum()) == 50                              # the final occurs once per run
    assert ((g["home_wins"] >= 0) & (g["home_wins"] <= g["n"])).all()
    assert (bracket["tournament"] == "T").all()
