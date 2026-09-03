"""Print row counts and a short summary of every persisted table.

The quickest way to see what state the data is in: what exists, how many rows, which
teams and dates it spans. A missing table is reported and skipped rather than raising,
since most of them are optional and none are in version control.

    python scripts/pipeline/inventory_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.utils.io import read_parquet    # noqa: E402

PROCESSED = Path("data/processed")

TABLES = [
    "matches", "team_ratings", "player_stats", "player_values", "team_style",
    "fixtures", "groups", "lineups", "injuries",
    "matchup_features", "match_predictions", "simulation_results", "bracket_results",
    "results_2026",
]
# Value counts on a high-cardinality column produce hundreds of lines of noise.
MAX_LISTED = 30


def summarize(name, df):
    print(f"=== {name}: {len(df):>7,} rows ===")
    print(f"     columns: {list(df.columns)}")
    if "team" in df.columns:
        teams = sorted(df["team"].unique())
        print(f"     unique teams: {len(teams)}")
        if len(teams) <= MAX_LISTED:
            print(f"     teams: {teams}")
    if "source" in df.columns:
        print(f"     sources: {df['source'].value_counts().to_dict()}")
    if "date" in df.columns:
        print(f"     dates: {df['date'].min()} .. {df['date'].max()}")
    if "season" in df.columns:
        print(f"     seasons: {sorted(df['season'].unique())[:15]}")
    if "player" in df.columns:
        print(f"     players: {df['player'].nunique():,}")
    if "competition" in df.columns:
        comps = df["competition"].value_counts()
        shown = comps.head(MAX_LISTED).to_dict()
        more = "" if len(comps) <= MAX_LISTED else f" (+{len(comps) - MAX_LISTED} more)"
        print(f"     competitions: {shown}{more}")
    print()


def main():
    missing = []
    for name in TABLES:
        path = PROCESSED / f"{name}.parquet"
        if not path.exists():
            missing.append(name)
            continue
        summarize(name, read_parquet(path))
    if missing:
        print("not present: " + ", ".join(missing))


if __name__ == "__main__":
    main()
