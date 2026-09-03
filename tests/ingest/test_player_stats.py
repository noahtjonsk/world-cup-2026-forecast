import pandas as pd
from src.schema import CANON_PLAYER_STAT_COLS
from src.ingest.player_stats import normalize_player_stats

def test_melts_wide_stats_to_long_canonical():
    wide = pd.DataFrame({
        "player": ["Rodri", "Bellingham"],
        "team": ["Man City", "Real Madrid"],
        "position": ["DM", "AM"],
        "npxg": [0.12, 0.41],
        "prog_passes": [8.3, 5.1],
    })
    out = normalize_player_stats(wide, source="fbref", season="2024-2025")
    assert list(out.columns) == CANON_PLAYER_STAT_COLS
    # 2 players x 2 metrics = 4 long rows
    assert len(out) == 4
    assert set(out["metric"]) == {"npxg", "prog_passes"}
    assert out.loc[out["player"] == "Rodri", "season"].iloc[0] == "2024-2025"
