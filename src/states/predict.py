import pandas as pd

FINISHED_STATUSES = {"FT", "AET", "PEN"}   # API-Football finished status codes

def upcoming_fixtures(fixtures, as_of):
    """Fixtures still to play as of `as_of`: status not finished AND kickoff on/after
    as_of. Returns the filtered frame (input columns preserved, index reset)."""
    df = fixtures.copy()
    df["date"] = pd.to_datetime(df["date"])
    mask = (~df["status"].isin(FINISHED_STATUSES)) & (df["date"] >= pd.Timestamp(as_of))
    return df[mask].reset_index(drop=True)

import numpy as np
from src.schema import CANON_PREDICTION_COLS
from src.utils.ids import make_match_id

def assemble_predictions(fixtures, wdl_probs, exp_goals, snapshot_date, source="live"):
    """Build a CANON_PREDICTION_COLS frame from upcoming `fixtures`, an (n,3) W/D/L
    array in fixed [H,D,A] order, and an (n,2) expected-goals array [home, away].
    Row i of each array aligns to fixtures.iloc[i]. match_id is the canonical
    make_match_id(date, home, away) so predictions join the matches/feature tables.
    Pure: model outputs are passed in, not computed here."""
    fx = fixtures.reset_index(drop=True)
    P = np.asarray(wdl_probs, dtype=float)
    G = np.asarray(exp_goals, dtype=float)
    rows = []
    for i, f in fx.iterrows():
        rows.append({
            "match_id": make_match_id(pd.Timestamp(f["date"]), f["home_team"], f["away_team"]),
            "date": pd.Timestamp(f["date"]),
            "home_team": f["home_team"], "away_team": f["away_team"],
            "snapshot_date": pd.Timestamp(snapshot_date),
            "p_home": float(P[i, 0]), "p_draw": float(P[i, 1]), "p_away": float(P[i, 2]),
            "exp_goals_home": float(G[i, 0]), "exp_goals_away": float(G[i, 1]),
            "source": source,
        })
    return pd.DataFrame(rows, columns=CANON_PREDICTION_COLS)
