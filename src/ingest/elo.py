import pandas as pd
from src.schema import CANON_RATING_COLS

def normalize_elo(raw, source="eloratings"):
    """Map a (date, team, elo) ratings frame to the canonical ratings schema."""
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["date"]),
        "team": raw["team"].astype(str),
        "elo": raw["elo"].astype(float),
    })
    df["source"] = source
    return df[CANON_RATING_COLS]

def fetch_clubelo():
    """Thin wrapper over the soccerdata ClubElo API: club Elo time series.
    For NATIONAL-team Elo use eloratings.net (download/scrape) then pass through
    normalize_elo with source='eloratings'."""
    from soccerdata import ClubElo
    return ClubElo().read_by_date()
