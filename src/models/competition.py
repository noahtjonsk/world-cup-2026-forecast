import numpy as np
import pandas as pd

def competition_weights(competition, tiers, default=0.2):
    """Per-match weight multiplier from free-text competition names. `tiers` is an
    ordered list of (substring, weight). A row matches a tier if the lower-cased
    substring is contained in the lower-cased competition name; the assigned weight
    is the MIN over all matching tiers, so any qualifier is <= its parent
    tournament's tier (e.g. "COSAFA Cup qualification" -> min(0.3, 0.8) = 0.3,
    "FIFA World Cup qualification" -> min(1.0, 0.8) = 0.8). NaN/None/unmatched ->
    `default`. Result clipped to >= 0. Vectorized over the Series; pure (numpy/pandas)."""
    s = pd.Series(competition).astype("string").str.lower()
    keys = [(str(k).lower(), float(w)) for k, w in tiers]
    base = max(float(default), 0.0)
    out = np.full(len(s), base, dtype=float)
    for i, name in enumerate(s):
        if pd.isna(name):
            continue
        matched = [w for k, w in keys if k and k in name]
        if matched:
            out[i] = max(min(matched), 0.0)
    return out
