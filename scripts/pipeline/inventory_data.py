"""Print row counts and a short summary of every persisted table.

The quickest way to see what state the data is in: what exists, how many rows,
which teams and dates it spans. Missing tables are reported rather than raising."""
from src.utils.io import read_parquet

files = {
    "matches": "data/processed/matches.parquet",
    "team_ratings": "data/processed/team_ratings.parquet",
    "player_stats": "data/processed/player_stats.parquet",
    "fixtures": "data/processed/fixtures.parquet",
    "lineups": "data/processed/lineups.parquet",
    "injuries": "data/processed/injuries.parquet",
    "player_values": "data/processed/player_values.parquet",
    "team_style": "data/processed/team_style.parquet",
}

for name, path in files.items():
    df = read_parquet(path)
    print(f"=== {name}: {len(df):>5} rows ===")
    print(f"     columns: {list(df.columns)}")
    if "team" in df.columns:
        teams = sorted(df["team"].unique())
        print(f"     unique teams: {len(teams)}")
        if len(teams) <= 30:
            print(f"     teams: {teams}")
    if "source" in df.columns:
        print(f"     sources: {df['source'].value_counts().to_dict()}")
    if "date" in df.columns:
        print(f"     dates: {df['date'].min()} .. {df['date'].max()}")
    if "season" in df.columns:
        print(f"     seasons: {sorted(df['season'].unique())[:15]}")
    if "player" in df.columns:
        print(f"     players: {df['player'].nunique()}")
    if "competition" in df.columns:
        print(f"     competitions: {df['competition'].value_counts().to_dict()}")
    print()
