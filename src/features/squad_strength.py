# src/features/squad_strength.py
import numpy as np
from src.features.player_vectors import pivot_player_metrics
from src.features.roles import assign_roles, to_in_role_percentiles, player_quality
from src.utils.ids import normalize_name

ATTACK_ROLES = {"ST", "W", "AM"}
DEFENCE_ROLES = {"GK", "CB", "FB", "DM"}

def _metric_cols(wide):
    return [c for c in wide.columns if c not in ("player", "position")]

def team_squad_strength(squads, player_stats, as_of, months=24,
                        n_att=4, n_def=5, min_players=3,
                        league_strength=None, include_midfield=False):
    """{nation: (atk_q, dfc_q)} from the role-aware quality of each nation's best
    players, leakage-safe as-of `as_of` (pivot_player_metrics routes through asof_window).

    Default (both opt-in params unset): atk_q = mean quality of top `n_att` ATTACK-role
    players; dfc_q = mean of top `n_def` DEFENCE-role players (CM excluded). Z-scored
    across nations, then shrunk by coverage fraction. Nations with < `min_players`
    covered -> (0.0, 0.0).

    Two optional refinements, both off by default:
    - league_strength: {club: multiplier} dict; when provided, each player's stat_quality
      is multiplied by their club's league multiplier before role selection.
    - include_midfield: when True, use a role-quota best XI (1 GK + 4 DEF{CB,FB} +
      3 MID{DM,CM,AM} + 3 ATK{W,ST}) and set atk_q = mean(ATK|MID), dfc_q = mean(GK,DEF|MID)
      so midfield counts toward both ends. The n_att/n_def params are ignored in this mode."""
    wide = pivot_player_metrics(player_stats, as_of, months=months)
    if wide.empty:
        return {n: (0.0, 0.0) for n in squads["nation"].unique()}
    mcols = _metric_cols(wide)
    quality = player_quality(to_in_role_percentiles(assign_roles(wide), mcols), mcols)
    quality["name_normalized"] = quality["player"].map(normalize_name)
    qmap = (quality.drop_duplicates("name_normalized")
            .set_index("name_normalized")[["role", "quality"]]
            .rename(columns={"role": "stat_role", "quality": "stat_quality"}))

    sq = squads.copy()
    if "name_normalized" not in sq.columns:
        sq["name_normalized"] = sq["player"].map(normalize_name)
    sq = sq.join(qmap, on="name_normalized")

    # Apply league multiplier before role selection (opt-in)
    if league_strength is not None and "club" in sq.columns:
        sq["stat_quality"] = sq["stat_quality"] * sq["club"].map(league_strength).fillna(1.0)

    raw, out = {}, {}
    if include_midfield:
        import pandas as pd
        # Role-quota best XI: 1 GK + 4 DEF{CB,FB} + 3 MID{DM,CM,AM} + 3 ATK{W,ST}
        GK_ROLES = {"GK"}
        DEF_ROLES_MID = {"CB", "FB"}
        MID_ROLES = {"DM", "CM", "AM"}
        ATK_ROLES_MID = {"W", "ST"}
        for nation, g in sq.groupby("nation"):
            covered = g.dropna(subset=["stat_quality"])
            if len(covered) < min_players:
                out[nation] = (0.0, 0.0)
                continue
            # Pick best per role bucket
            best_gk = covered[covered["stat_role"].isin(GK_ROLES)].nlargest(1, "stat_quality")["stat_quality"]
            best_def = covered[covered["stat_role"].isin(DEF_ROLES_MID)].nlargest(4, "stat_quality")["stat_quality"]
            best_mid = covered[covered["stat_role"].isin(MID_ROLES)].nlargest(3, "stat_quality")["stat_quality"]
            best_atk = covered[covered["stat_role"].isin(ATK_ROLES_MID)].nlargest(3, "stat_quality")["stat_quality"]
            # atk_q = mean(ATK + MID), dfc_q = mean(GK + DEF + MID)
            atk_pool = pd.concat([best_atk, best_mid])
            dfc_pool = pd.concat([best_gk, best_def, best_mid])
            atk = float(atk_pool.mean()) if len(atk_pool) else float(covered["stat_quality"].mean())
            dfc = float(dfc_pool.mean()) if len(dfc_pool) else float(covered["stat_quality"].mean())
            raw[nation] = (atk, dfc, len(covered) / len(g))
    else:
        for nation, g in sq.groupby("nation"):
            covered = g.dropna(subset=["stat_quality"])
            if len(covered) < min_players:
                out[nation] = (0.0, 0.0)
                continue
            att = covered[covered["stat_role"].isin(ATTACK_ROLES)].nlargest(n_att, "stat_quality")["stat_quality"]
            dfn = covered[covered["stat_role"].isin(DEFENCE_ROLES)].nlargest(n_def, "stat_quality")["stat_quality"]
            atk = float(att.mean()) if len(att) else float(covered["stat_quality"].mean())
            dfc = float(dfn.mean()) if len(dfn) else float(covered["stat_quality"].mean())
            raw[nation] = (atk, dfc, len(covered) / len(g))
    if not raw:
        return {n: out.get(n, (0.0, 0.0)) for n in squads["nation"].unique()}
    atks = np.array([v[0] for v in raw.values()])
    dfcs = np.array([v[1] for v in raw.values()])
    a_mu, a_sd = atks.mean(), (atks.std() or 1.0)
    d_mu, d_sd = dfcs.mean(), (dfcs.std() or 1.0)
    for nation, (atk, dfc, cov) in raw.items():
        out[nation] = (cov * (atk - a_mu) / a_sd, cov * (dfc - d_mu) / d_sd)
    return out


def club_league_multiplier(clubs, ls_path="data/reference/league_strength.csv",
                           cl_path="data/reference/club_league.csv", default=0.45):
    """{club: multiplier} for the given clubs. Joins club->league (cl_path) to
    league->multiplier (ls_path); clubs with no/unknown league -> `default`. Pure
    apart from reading the two committed reference CSVs."""
    import pandas as pd
    ls = pd.read_csv(ls_path)
    cl = pd.read_csv(cl_path)
    lm = dict(zip(ls["league"], ls["multiplier"]))
    club2league = dict(zip(cl["club"], cl["league"]))
    return {c: float(lm.get(club2league.get(c), default)) for c in clubs}
