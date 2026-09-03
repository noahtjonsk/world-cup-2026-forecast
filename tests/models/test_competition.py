import pandas as pd
from src.models.competition import competition_weights

TIERS = [("FIFA World Cup", 1.0), ("World Cup 20", 1.0), ("AFC Asian Cup", 1.0),
         ("qualification", 0.8), ("Gulf Cup", 0.5), ("COSAFA", 0.3), ("Friendly", 0.2)]

def test_exact_substring_and_default():
    s = pd.Series(["Friendly", "FIFA World Cup", "Gulf Cup", "Mystery League", None])
    w = competition_weights(s, TIERS, default=0.2)
    assert list(w) == [0.2, 1.0, 0.5, 0.2, 0.2]      # unmatched + None -> default

def test_min_weight_among_matches():
    # min-weight rule: a qualifier is <= its parent tier; a minor-cup qualifier stays minor
    s = pd.Series(["FIFA World Cup qualification", "AFC Asian Cup qualification",
                   "COSAFA Cup qualification", "World Cup 2018"])
    w = competition_weights(s, TIERS, default=0.2)
    assert list(w) == [0.8, 0.8, 0.3, 1.0]

def test_case_insensitive_and_float_dtype():
    w = competition_weights(pd.Series(["fifa world cup", "FRIENDLY"]), TIERS, default=0.2)
    assert w.dtype == float and list(w) == [1.0, 0.2]

def test_negative_default_clipped_nonnegative():
    w = competition_weights(pd.Series(["x"]), TIERS, default=-1.0)
    assert w[0] == 0.0

def test_mojibake_copa_america_matches_ascii_substring():
    w = competition_weights(pd.Series(["Copa Am�rica"]), [("Copa Am", 1.0)], default=0.2)
    assert w[0] == 1.0
