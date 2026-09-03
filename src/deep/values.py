import pandas as pd
from src.schema import CANON_PLAYER_STAT_COLS, CANON_TEAM_STYLE_COLS

# raw VAEP/xT column -> per-90 metric name
VALUE_COLS = {
    "vaep_value": "vaep_p90",
    "offensive_value": "vaep_off_p90",
    "defensive_value": "vaep_def_p90",
    "xt_value": "xt_p90",
}

def aggregate_player_values(actions, players, season, min_minutes=180, source="vaep"):
    """Sum action values per player, normalise to per-90, drop low-minutes players,
    return long player_stats rows. (Recipe per socceraction's verified aggregation.)"""
    sums = (actions.groupby(["player", "team"], as_index=False)[list(VALUE_COLS)].sum())
    sums = sums.merge(players[["player", "minutes_played"]], on="player", how="left")
    sums = sums[sums["minutes_played"] > min_minutes].copy()
    for raw, name in VALUE_COLS.items():
        sums[name] = sums[raw] * 90 / sums["minutes_played"]
    long = sums.melt(id_vars=["player", "team"], value_vars=list(VALUE_COLS.values()),
                     var_name="metric", value_name="value")
    long["season"] = str(season)
    long["position"] = None
    long["source"] = source
    return long[CANON_PLAYER_STAT_COLS]

def decompose_vaep_by_action_type(actions, players, season, min_minutes=180, source="vaep_by_type"):
    """Per-player VAEP per 90 split by action type, as metrics like 'vaep_pass_p90'.

    An open-data stand-in for on-ball value broken down by what the player actually
    did, rather than a single number per player."""
    g = (actions.groupby(["player", "team", "action_type"], as_index=False)["vaep_value"].sum())
    g = g.merge(players[["player", "minutes_played"]], on="player", how="left")
    g = g[g["minutes_played"] > min_minutes].copy()
    g["value"] = g["vaep_value"] * 90 / g["minutes_played"]
    g["metric"] = "vaep_" + g["action_type"].astype(str) + "_p90"
    g["season"] = str(season)
    g["position"] = None
    g["source"] = source
    return g[CANON_PLAYER_STAT_COLS]

def compute_team_style(actions, season, source="vaep"):
    """Team tactical fingerprint (long): action-type shares + mean xT per action."""
    total = actions.groupby("team").size().rename("n")
    by_type = actions.groupby(["team", "action_type"]).size().rename("n_type").reset_index()
    by_type = by_type.merge(total, on="team")
    by_type["value"] = by_type["n_type"] / by_type["n"]
    by_type["metric"] = "share_" + by_type["action_type"].astype(str)
    shares = by_type[["team", "metric", "value"]]

    mxt = actions.groupby("team", as_index=False)["xt_value"].mean().rename(columns={"xt_value": "value"})
    mxt["metric"] = "mean_xt"

    style = pd.concat([shares, mxt[["team", "metric", "value"]]], ignore_index=True)
    style["season"] = str(season)
    style["source"] = source
    return style[CANON_TEAM_STYLE_COLS]

def assemble_deep_tables(actions, players, season, min_minutes=180):
    """Build the deep-tier output tables, ready for persist_tables()."""
    player_values = pd.concat([
        aggregate_player_values(actions, players, season, min_minutes),
        decompose_vaep_by_action_type(actions, players, season, min_minutes),
    ], ignore_index=True)
    team_style = compute_team_style(actions, season)
    return {"player_values": player_values, "team_style": team_style}
