import pandas as pd
from src.schema import CANON_RATING_COLS
from src.ingest.elo import normalize_elo

def test_normalize_elo_maps_to_canonical():
    raw = pd.DataFrame({"date": ["2026-03-01", "2026-03-01"],
                        "team": ["Brazil", "France"],
                        "elo": ["2100", "2050"]})   # strings on purpose
    out = normalize_elo(raw, source="eloratings")
    assert list(out.columns) == CANON_RATING_COLS
    assert out["elo"].dtype == float
    assert out.loc[0, "source"] == "eloratings"
