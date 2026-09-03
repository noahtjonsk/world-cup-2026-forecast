import pandas as pd

REQUIRED_ROLES = ("GK", "CB", "FB", "DM", "CM", "AM", "W", "ST")

def squad_features(lineup_rows, quality_df, required_roles=REQUIRED_ROLES):
    """Aggregate per-player role quality over a match lineup into per-team indices.

    lineup_rows: one match's rows with columns team, player, is_starter.
    quality_df:  player, role, quality (in-role percentile quality, 0..1).
    Returns {team: {xi_quality, bench_dropoff, role_coverage}}.
      - xi_quality:    mean quality of starters.
      - bench_dropoff: xi_quality - mean quality of bench (NaN if no bench).
      - role_coverage: mean over required_roles of the best starter in that role
                       (0 for an unfilled role) -> rewards a complete, strong XI.

    Players absent from quality_df are NaN after the left merge: they are skipped in
    the quality means (xi_quality / bench), and a role whose only starters are unscored
    counts as 0 in role_coverage (same as an unfilled role)."""
    j = lineup_rows.merge(quality_df, on="player", how="left")
    out = {}
    for team, g in j.groupby("team"):
        starters = g[g["is_starter"]]
        bench = g[~g["is_starter"]]
        xi = float(starters["quality"].mean()) if len(starters) else float("nan")
        bench_q = float(bench["quality"].mean()) if len(bench) else float("nan")
        covers = []
        for r in required_roles:
            best = starters.loc[starters["role"] == r, "quality"].max()  # skips NaN
            covers.append(0.0 if pd.isna(best) else float(best))         # absent OR unscored -> 0
        out[team] = {
            "xi_quality": xi,
            "bench_dropoff": (xi - bench_q) if not pd.isna(bench_q) else float("nan"),
            "role_coverage": float(pd.Series(covers).mean()),
        }
    return out
