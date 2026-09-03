from src.simulation.bracket import next_round, seed_knockout


def test_next_round_pairs_consecutive_winners():
    assert next_round(["W1", "W2", "W3", "W4"]) == [("W1", "W2"), ("W3", "W4")]
    assert next_round(["X", "Y"]) == [("X", "Y")]


def test_seed_knockout_resolves_tokens():
    rankings = {"A": ["A1", "A2", "A3", "A4"], "B": ["B1", "B2", "B3", "B4"]}
    thirds = [("C", "C3"), ("D", "D3")]                 # ranked best-third list
    seeding = [["1A", "2B"], ["1B", "T1"], ["2A", "T2"]]
    assert seed_knockout(rankings, thirds, seeding) == [
        ("A1", "B2"), ("B1", "C3"), ("A2", "D3"),
    ]
