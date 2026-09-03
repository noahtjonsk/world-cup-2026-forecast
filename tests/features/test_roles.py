# tests/features/test_roles.py
import pandas as pd
from src.features.roles import (
    position_to_role, assign_roles, to_in_role_percentiles, player_quality, cluster_roles,
)

def test_position_to_role_buckets():
    assert position_to_role("CB") == "CB"
    assert position_to_role("LWB") == "FB"
    assert position_to_role("CAM") == "AM"
    assert position_to_role("ST") == "ST"
    assert position_to_role(None) == "UNK"
    assert position_to_role(float("nan")) == "UNK"      # pandas NaN, not None
    # coarse codes the live sources emit (FBref/Understat/API-Football) collapse to buckets
    assert position_to_role("DF") == "CB"
    assert position_to_role("M") == "CM"
    assert position_to_role("F") == "ST"
    assert position_to_role("G") == "GK"
    assert position_to_role("DF,MF") == "CB"            # combo -> primary token
    assert position_to_role("Zog") == "UNK"             # genuinely unknown

def test_roles_assign_percentiles_quality():
    pv = pd.DataFrame({
        "player": ["A", "B", "C", "D"],
        "position": ["ST", "ST", "CB", "CB"],
        "vaep_p90": [0.9, 0.3, 0.5, 0.1],
    })
    roled = assign_roles(pv)
    assert list(roled["role"]) == ["ST", "ST", "CB", "CB"]
    pct = to_in_role_percentiles(roled, ["vaep_p90"])
    vals = dict(zip(pct["player"], pct["vaep_p90"]))
    # within ST: 0.9 -> 1.0, 0.3 -> 0.5 ; within CB: 0.5 -> 1.0, 0.1 -> 0.5
    assert (vals["A"], vals["B"], vals["C"], vals["D"]) == (1.0, 0.5, 1.0, 0.5)
    q = player_quality(pct, ["vaep_p90"])
    assert abs(q.loc[q.player == "A", "quality"].iloc[0] - 1.0) < 1e-9
    assert callable(cluster_roles)   # optional thin wrapper present (scikit-learn, not unit-run)
