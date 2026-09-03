import pandas as pd
from src.schema import CANON_MATCH_COLS
from src.ingest.statsbomb import normalize_matches

def test_normalize_maps_statsbombpy_columns_to_canonical():
    raw = pd.DataFrame({
        "match_date": ["2018-06-14"],
        "competition": ["FIFA World Cup"],
        "season": ["2018"],
        "home_team": ["Russia"],
        "away_team": ["Saudi Arabia"],
        "home_score": [5],
        "away_score": [0],
        "competition_stage": ["Group Stage"],
    })
    out = normalize_matches(raw)
    assert list(out.columns) == CANON_MATCH_COLS
    assert out.loc[0, "source"] == "statsbomb"
    assert out.loc[0, "neutral"] == True
    assert len(out.loc[0, "match_id"]) == 16
