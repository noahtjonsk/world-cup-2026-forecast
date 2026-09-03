import pandas as pd

STAGE_CODE = {
    "group": 0, "group stage": 0,
    "round of 32": 1, "r32": 1,
    "round of 16": 2, "r16": 2,
    "quarter-finals": 3, "quarterfinals": 3, "qf": 3,
    "semi-finals": 4, "semifinals": 4, "sf": 4,
    "third place": 4,   # same tier as SF, the 3rd-place playoff is not a deeper stage
    "final": 5,
}

def stage_code(stage):
    """Ordinal tournament-stage code (group=0 ... final=5). Unknown/None -> 0."""
    if stage is None:
        return 0
    return STAGE_CODE.get(str(stage).strip().lower(), 0)

def rest_days(matches, team, kickoff):
    """Days since the team's most recent match strictly before kickoff (NaN if none)."""
    m = matches[(matches["home_team"] == team) | (matches["away_team"] == team)].copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] < pd.Timestamp(kickoff)]
    if m.empty:
        return float("nan")
    return float((pd.Timestamp(kickoff) - m["date"].max()).days)

def context_features(matches, match_row, host_teams=()):
    """Rest / stage / neutral / host features for one match row -> dict.

    `host_teams` must use the same team names as `matches`. Host flags are an exact
    string match, so a name that does not line up yields 0 without complaining. That
    is how the United States went a whole build without home advantage, matched as
    "USA" against a table that calls it "United States"."""
    home, away, kickoff = match_row["home_team"], match_row["away_team"], match_row["date"]
    hosts = {str(t) for t in host_teams}
    rh = rest_days(matches, home, kickoff)
    ra = rest_days(matches, away, kickoff)
    neutral = match_row.get("neutral")  # pd.isna handles None / NaN / pd.NA uniformly
    return {
        "rest_home": rh, "rest_away": ra, "rest_diff": rh - ra,
        "stage_code": stage_code(match_row.get("stage")),
        "neutral": 0 if pd.isna(neutral) else int(bool(neutral)),
        "host_home": int(home in hosts),
        "host_away": int(away in hosts),
    }
