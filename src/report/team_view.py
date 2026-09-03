import pandas as pd


def style_profile(team_style, team, season=None):
    """Per-metric style profile [metric, value] for one team from the long
    team_style table (CANON_TEAM_STYLE_COLS). Defaults to the latest season."""
    df = team_style[team_style["team"] == team]
    if season is not None:
        df = df[df["season"] == season]
    elif not df.empty:
        df = df[df["season"] == df["season"].max()]
    return df[["metric", "value"]].sort_values("metric").reset_index(drop=True)


def strength_summary(ratings, features, team):
    """{team, elo, xi_quality, role_coverage} from the latest Elo (ratings,
    CANON_RATING_COLS) and the most-recent matchup_features row involving the team
    (home or away side resolved). Missing inputs -> None. Read-only."""
    r = ratings[ratings["team"] == team].copy()
    r["date"] = pd.to_datetime(r["date"])
    r = r.sort_values("date")
    elo = float(r["elo"].iloc[-1]) if not r.empty else None

    f = features[(features["home_team"] == team) | (features["away_team"] == team)].copy()
    xi = role = None
    if not f.empty:
        f["date"] = pd.to_datetime(f["date"])
        last = f.sort_values("date").iloc[-1]
        side = "home" if last["home_team"] == team else "away"
        xi = float(last[f"xi_quality_{side}"])
        role = float(last[f"role_coverage_{side}"])
    return {"team": team, "elo": elo, "xi_quality": xi, "role_coverage": role}
