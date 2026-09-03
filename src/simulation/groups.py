import re
import pandas as pd
from src.schema import CANON_GROUP_COLS


def parse_groups(fixtures, tournament):
    """Extract {tournament, group, team} rows from a fixtures frame whose `stage`
    labels group-stage matches like 'Group A'. Teams come from group-stage fixtures
    only; the group letter (A-L) is parsed from the stage text. Returns
    CANON_GROUP_COLS, deduplicated."""
    rows = []
    for _, fx in fixtures.iterrows():
        m = re.search(r"group\s+([a-l])\b", str(fx.get("stage", "")), re.IGNORECASE)
        if not m:
            continue
        g = m.group(1).upper()
        for team in (fx["home_team"], fx["away_team"]):
            rows.append({"tournament": tournament, "group": g, "team": team})
    if not rows:
        return pd.DataFrame(columns=CANON_GROUP_COLS)
    return pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)[CANON_GROUP_COLS]


def groups_to_dict(groups_df):
    """CANON_GROUP_COLS frame -> {group: [team, ...]} for the simulator."""
    return {g: list(sub["team"]) for g, sub in groups_df.groupby("group")}
