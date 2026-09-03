import pandas as pd
from src.utils.dates import asof_window

def test_keeps_only_in_window_and_pre_kickoff():
    df = pd.DataFrame({"date": ["2023-01-01", "2024-08-01", "2026-06-01", "2026-07-01"]})
    out = asof_window(df, kickoff="2026-06-05", months=24)
    kept = sorted(pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d"))
    # 2023-01-01 too old (>24mo); 2026-07-01 after kickoff -> both dropped
    assert kept == ["2024-08-01", "2026-06-01"]
