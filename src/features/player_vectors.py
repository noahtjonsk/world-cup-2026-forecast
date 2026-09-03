# src/features/player_vectors.py
import pandas as pd
from src.utils.dates import asof_window

def season_end_date(season):
    """Date a season's stats become fully known (season end). Conservative for leakage:
    '2024-2025' -> 2025-06-30 ; '2024' -> 2024-12-31. Also accepts FBref's native
    2-digit end form ('2024-25' -> 2025-06-30) so a season is never silently misdated."""
    s = str(season)
    if "-" in s:
        start, end = s.split("-")
        if int(start) <= 100:  # century-carry below assumes a 4-digit start year
            raise ValueError(f"season start year must be 4-digit: {season!r}")
        end = int(end)
        if end < 100:  # 2-digit end (e.g. '2024-25'); carry the start year's century
            end += int(start) - int(start) % 100
        return pd.Timestamp(end, 6, 30)
    return pd.Timestamp(int(s), 12, 31)

def pivot_player_metrics(player_stats, kickoff, months=24):
    """Long `player_stats` -> wide per-player metric table, leakage-safe by season.

    Synthesises `date = season_end_date(season)`, routes through `asof_window` (the
    single leakage chokepoint), pivots metric->columns (mean over rows), and carries
    each player's most common non-null position. Returns a `player`/`position` frame
    even when empty."""
    df = player_stats.copy()
    df["date"] = df["season"].map(season_end_date)
    df = asof_window(df, kickoff, months=months)
    if df.empty:
        return pd.DataFrame(columns=["player", "position"])
    wide = (df.pivot_table(index="player", columns="metric", values="value", aggfunc="mean")
              .reset_index())
    wide.columns.name = None
    # mode() returns its values sorted, so iloc[0] breaks ties on the alphabetically
    # first position; players with all-null positions fall through to NaN via the merge.
    pos = (df.dropna(subset=["position"]).groupby("player")["position"]
             .agg(lambda s: s.mode().iloc[0]).rename("position"))
    return wide.merge(pos, on="player", how="left")
