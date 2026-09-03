import numpy as np
from src.simulation.tournament import simulate_tournament


def _two_group_final_format():
    return {"qualifiers_per_group": 1, "third_place_slots": 0, "rounds": ["F"]}


def test_two_group_final_winner_and_reached():
    fmt = _two_group_final_format()
    groups = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    params = {"attack": {"A1": 2.0, "A2": -2.0, "B1": 2.0, "B2": -2.0},
              "defence": {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0},
              "home_adv": 0.0, "rho": 0.0}
    seeding = [["1A", "1B"]]                                    # group winners meet in the final
    champ, reached = simulate_tournament(fmt, groups, params, seeding, np.random.default_rng(0))
    assert champ in {"A1", "B1"}                                # a strong group winner won
    assert reached[champ] == 2                                  # F=stage1, champion=stage2
    assert reached["A2"] == 0 and reached["B2"] == 0            # eliminated in the group
    assert reached["A1"] >= 1 and reached["B1"] >= 1            # both reached the final


def test_bracket_log_records_knockout_matches():
    fmt = _two_group_final_format()
    groups = {"A": ["A1", "A2"], "B": ["B1", "B2"]}
    params = {"attack": {"A1": 2.0, "A2": -2.0, "B1": 2.0, "B2": -2.0},
              "defence": {"A1": 0.0, "A2": 0.0, "B1": 0.0, "B2": 0.0},
              "home_adv": 0.0, "rho": 0.0}
    log = []
    champ, _ = simulate_tournament(fmt, groups, params, [["1A", "1B"]],
                                   np.random.default_rng(0), bracket_log=log)
    assert len(log) == 1                                        # one knockout match (the final)
    rnd, idx, home, away, winner = log[0]
    assert rnd == "F" and idx == 0
    assert (home, away) == ("A1", "B1") or (home, away) == ("B1", "A1")  # group winners
    assert winner == champ
