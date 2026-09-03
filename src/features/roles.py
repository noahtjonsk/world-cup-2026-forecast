# src/features/roles.py
import re
import pandas as pd

# nominal position code -> role bucket.
# Granular codes (CB/DM/AM...) come from sources that supply them; the coarse codes
# (FBref DF/MF/FW, Understat D/M/F/S, API-Football G/D/M/F) are the common case and
# collapse onto the existing buckets when finer granularity is unavailable, a generic
# defender lands in CB, a generic midfielder in CM, a generic forward in ST. This keeps
# the in-role percentile pools sane (FBref/Understat are the dominant player_stats source)
# rather than dumping most players into 'UNK'.
# NOTE: StatsBomb full-name positions ("Right Center Back") must be normalised to these
# codes upstream; deep-tier VAEP rows carry no position (-> 'UNK'). Anything unmapped
# falls to 'UNK'.
ROLE_MAP = {
    "GK": "GK", "G": "GK",
    "CB": "CB", "RCB": "CB", "LCB": "CB", "D": "CB", "DF": "CB", "DEF": "CB",
    "RB": "FB", "LB": "FB", "RWB": "FB", "LWB": "FB",
    "DM": "DM", "CDM": "DM",
    "CM": "CM", "LCM": "CM", "RCM": "CM", "MF": "CM", "M": "CM", "MID": "CM",
    "AM": "AM", "CAM": "AM",
    "RM": "W", "LM": "W", "RW": "W", "LW": "W", "W": "W",
    "CF": "ST", "ST": "ST", "SS": "ST", "FW": "ST", "F": "ST", "S": "ST", "FWD": "ST",
}

def position_to_role(position):
    """Bucket a nominal position code into a role group. None/NaN/unknown -> 'UNK'.

    Falls back to the primary (first) token for combo codes like 'DF,MF' or 'FW MF'."""
    if position is None:
        return "UNK"
    code = str(position).upper().strip()
    if code in ROLE_MAP:
        return ROLE_MAP[code]
    first = re.split(r"[,/ ]+", code)[0]   # FBref-style combos -> primary position
    return ROLE_MAP.get(first, "UNK")

def assign_roles(player_vectors, position_col="position"):
    """Add a `role` column via `position_to_role`. (Data-driven clustering refinement
    lives in `cluster_roles`, an optional thin wrapper.)"""
    out = player_vectors.copy()
    out["role"] = out[position_col].map(position_to_role)
    return out

def to_in_role_percentiles(player_vectors, metric_cols, role_col="role"):
    """Rank each metric WITHIN role -> percentile in [0, 1] (smarterscout-style).
    Replaces each metric column with its in-role percentile."""
    out = player_vectors.copy()
    for m in metric_cols:
        # a role pool of size 1 always yields percentile 1.0 (pandas rank semantics)
        out[m] = out.groupby(role_col)[m].rank(pct=True)
    return out

def player_quality(percentile_vectors, metric_cols):
    """Collapse a player's in-role metric percentiles to one quality score (mean
    percentile, skipping missing metrics). Returns player, role, quality."""
    out = percentile_vectors.copy()
    out["quality"] = out[metric_cols].mean(axis=1)
    return out[["player", "role", "quality"]]

def cluster_roles(metric_matrix, n_roles=8, seed=42):
    """Derive role labels from the data by clustering the player metric matrix.

    An alternative to the fixed position-code mapping above: instead of trusting the
    listed position, group players by how they actually play. Not fixture-tested,
    since scikit-learn is an optional dependency.

    Usage: cluster_roles(wide[metric_cols].fillna(0.0).to_numpy(), n_roles=8)."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_roles, random_state=seed, n_init=10).fit_predict(metric_matrix)
