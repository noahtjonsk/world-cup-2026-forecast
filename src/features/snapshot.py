# src/features/snapshot.py
import pandas as pd
from src.schema import CANON_FEATURE_COLS
from src.features.strength import strength_features
from src.features.player_vectors import pivot_player_metrics
from src.features.roles import assign_roles, to_in_role_percentiles, player_quality
from src.features.squad import squad_features
from src.features.style import style_mismatch
from src.features.context import context_features

def match_result(home_score, away_score):
    """W/D/L label from the home team's perspective: 'H'/'D'/'A'; None if unplayed."""
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    if home_score > away_score:
        return "H"
    if home_score < away_score:
        return "A"
    return "D"

def _metric_cols(wide):
    return [c for c in wide.columns if c not in ("player", "position", "role")]

def _empty_squad():
    return {"xi_quality": float("nan"), "bench_dropoff": float("nan"), "role_coverage": float("nan")}

def build_features(match_id, matches, ratings, player_stats, team_style, lineups,
                   months=24, host_teams=(), snapshot_date=None):
    """One leakage-safe, pre-kickoff feature row (CANON_FEATURE_COLS) for a match.

    `lineups` must be keyed by `match_id` (the matches join key). Historical StatsBomb
    lineups are match_id-keyed; live API-Football lineups are `fixture_id`-keyed
    (CANON_LINEUP_COLS) and must be re-keyed to match_id by the caller before reaching
    here. A non-empty lineups frame lacking a `match_id` column is a wiring bug and
    raises; pass an empty frame for fixtures whose XI is not yet known.

    Squad features come through as NaN when no lineup rows exist for the match, which
    is the normal state for a future fixture before its predicted eleven is built.
    All as-of filtering happens inside the transforms this calls, via `asof_window`,
    so no caller has to remember to do it."""
    hits = matches.loc[matches["match_id"] == match_id]
    if hits.empty:
        raise KeyError(f"match_id {match_id!r} not found in matches table")
    row = hits.iloc[0]
    kickoff = row["date"]
    home, away, season = row["home_team"], row["away_team"], row["season"]

    feat = {
        "match_id": match_id,
        "date": pd.Timestamp(kickoff),
        "home_team": home,
        "away_team": away,
        "snapshot_date": pd.Timestamp(snapshot_date) if snapshot_date is not None else pd.Timestamp(kickoff),
    }
    feat.update(strength_features(matches, ratings, row, months=months))
    feat.update(style_mismatch(team_style, home, away, season))
    feat.update(context_features(matches, row, host_teams=host_teams))

    sq_home, sq_away = _empty_squad(), _empty_squad()
    if len(lineups) and "match_id" not in lineups.columns:
        raise KeyError(
            "lineups must be keyed by 'match_id' (re-key fixture_id->match_id upstream); "
            "pass an empty frame for fixtures without a known XI"
        )
    lm = lineups[lineups["match_id"] == match_id] if "match_id" in lineups.columns else lineups.iloc[0:0]
    if len(lm):
        wide = pivot_player_metrics(player_stats, kickoff, months=months)
        mcols = _metric_cols(wide)
        if len(wide) and mcols:
            from src.utils.ids import normalize_name
            quality = player_quality(to_in_role_percentiles(assign_roles(wide), mcols), mcols)
            # normalize both sides of the lineup->quality join: lineups carry Wikipedia
            # spellings, stats carry FBref spellings (raw equality loses ~6% of players)
            quality = quality.assign(player=quality["player"].map(normalize_name))
            quality = quality.drop_duplicates("player")
            lmn = lm[["team", "player", "is_starter"]].assign(
                player=lm["player"].map(normalize_name))
            sq = squad_features(lmn, quality)
            sq_home = sq.get(home, _empty_squad())
            sq_away = sq.get(away, _empty_squad())
    feat.update({
        "xi_quality_home": sq_home["xi_quality"],
        "xi_quality_away": sq_away["xi_quality"],
        # NaN if either side's lineup is absent, asymmetric data is treated as unknown
        "xi_quality_diff": sq_home["xi_quality"] - sq_away["xi_quality"],
        "bench_dropoff_home": sq_home["bench_dropoff"],
        "bench_dropoff_away": sq_away["bench_dropoff"],
        "role_coverage_home": sq_home["role_coverage"],
        "role_coverage_away": sq_away["role_coverage"],
    })
    feat["result"] = match_result(row.get("home_score"), row.get("away_score"))
    return pd.DataFrame([feat])[CANON_FEATURE_COLS]

def build_feature_table(match_ids, matches, ratings, player_stats, team_style, lineups,
                        months=24, host_teams=(), snapshot_date=None):
    """Build one feature row per match_id and stack into the `matchup_features` table
    (CANON_FEATURE_COLS). Pass the result to `persist_tables({'matchup_features': df})`."""
    rows = [build_features(mid, matches, ratings, player_stats, team_style, lineups,
                           months=months, host_teams=host_teams, snapshot_date=snapshot_date)
            for mid in match_ids]
    return pd.concat(rows, ignore_index=True)[CANON_FEATURE_COLS]
