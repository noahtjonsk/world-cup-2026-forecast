# tests/features/test_player_vectors.py
import pandas as pd
from src.features.player_vectors import season_end_date, pivot_player_metrics

def test_season_end_date_split_and_single_year():
    assert season_end_date("2024-2025") == pd.Timestamp(2025, 6, 30)
    assert season_end_date("2024") == pd.Timestamp(2024, 12, 31)
    assert season_end_date("2024-25") == pd.Timestamp(2025, 6, 30)   # FBref 2-digit end form


def test_pivot_player_metrics_empty_returns_position_column():
    # All rows after kickoff -> empty frame, but the downstream contract (assign_roles)
    # still needs a `position` column present.
    ps = pd.DataFrame({
        "player": ["X"], "team": ["Y"], "season": ["2030-2031"], "position": ["ST"],
        "metric": ["vaep_p90"], "value": [1.0], "source": ["vaep"],
    })
    wide = pivot_player_metrics(ps, kickoff="2026-06-01", months=24)
    assert wide.empty
    assert list(wide.columns) == ["player", "position"]

def test_pivot_player_metrics_asof_by_season():
    ps = pd.DataFrame({
        "player": ["Rodri", "Rodri", "Pedri", "Pedri"],
        "team":   ["Man City", "Spain", "Barcelona", "Spain"],
        "season": ["2024-2025", "2024-2025", "2024-2025", "2030-2031"],  # last is FUTURE season
        "position": ["DM", "DM", "CM", "CM"],
        "metric": ["vaep_p90"] * 4,
        "value":  [0.5, 0.3, 0.4, 99.0],
        "source": ["vaep", "vaep", "fbref", "fbref"],
    })
    wide = pivot_player_metrics(ps, kickoff="2026-06-01", months=24)
    assert set(wide["player"]) == {"Rodri", "Pedri"}                 # 2030-2031 excluded
    assert abs(wide.loc[wide.player == "Rodri", "vaep_p90"].iloc[0] - 0.4) < 1e-9  # mean(0.5,0.3)
    assert abs(wide.loc[wide.player == "Pedri", "vaep_p90"].iloc[0] - 0.4) < 1e-9  # future row dropped
    assert {"player", "vaep_p90", "position"} <= set(wide.columns)
