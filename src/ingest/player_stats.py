import pandas as pd
from src.schema import CANON_PLAYER_STAT_COLS

def normalize_player_stats(raw, source, season, id_cols=("player", "team", "position")):
    """Melt a wide per-player stats frame into the long canonical schema.

    Any source (FBref, Understat) and any metric set fits one table. Recency
    (24-month window) is applied later at feature-build time, not here.
    """
    id_cols = list(id_cols)
    value_cols = [c for c in raw.columns if c not in id_cols]
    long = raw.melt(id_vars=id_cols, value_vars=value_cols,
                    var_name="metric", value_name="value")
    long["season"] = str(season)
    long["source"] = source
    return long[CANON_PLAYER_STAT_COLS]

def fetch_fbref(league, season, stat_type="standard"):
    """Thin wrapper. Flatten FBref's MultiIndex columns to a
    simple wide frame (player, team, position, <metrics>) before normalize."""
    from soccerdata import FBref
    return FBref(leagues=league, seasons=season).read_player_season_stats(stat_type=stat_type)

def fetch_understat(league, season):
    """Thin wrapper: Understat player season stats (xG, xA, xGChain, xGBuildup)."""
    from soccerdata import Understat
    return Understat(leagues=league, seasons=season).read_player_season_stats()
