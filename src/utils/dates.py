import pandas as pd

def asof_window(df, kickoff, months=24, date_col="date"):
    """Return rows strictly before kickoff AND no older than `months`.

    Enforces the project's two leakage rules in one place: nothing after
    kickoff, nothing outside the rolling recency window.
    """
    kickoff = pd.Timestamp(kickoff)
    lower = kickoff - pd.DateOffset(months=months)
    dates = pd.to_datetime(df[date_col])
    mask = (dates < kickoff) & (dates >= lower)
    return df.loc[mask].copy()
