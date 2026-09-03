import pandas as pd
from src.schema import CANON_MATCH_COLS
from src.utils.ids import make_match_id

def normalize_matches(raw):
    """Map a statsbombpy matches DataFrame to the canonical schema (pure)."""
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["match_date"]),
        "competition": raw["competition"].astype(str),
        "season": raw["season"].astype(str),
        "home_team": raw["home_team"].astype(str),
        "away_team": raw["away_team"].astype(str),
        "home_score": raw["home_score"].astype("Int64"),
        "away_score": raw["away_score"].astype("Int64"),
        "stage": raw.get("competition_stage", pd.Series([None] * len(raw))).astype("object"),
    })
    df["neutral"] = True            # international tournaments default to neutral venues
    df["source"] = "statsbomb"
    df["match_id"] = [make_match_id(d, h, a)
                      for d, h, a in zip(df["date"], df["home_team"], df["away_team"])]
    return df[CANON_MATCH_COLS]

def fetch_matches(competition_id, season_id):
    """Thin network wrapper (not unit-tested). E.g. 43/3 = Men's WC 2018."""
    from statsbombpy import sb
    return sb.matches(competition_id=competition_id, season_id=season_id)
