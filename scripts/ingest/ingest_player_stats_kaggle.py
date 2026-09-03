"""Load the Kaggle FBref player statistics into the canonical player_stats table.

Reshapes the source CSVs into the long format the rest of the project expects,
one row per player, metric and season, so new sources can be appended without
schema changes."""
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, ".")
from src.ingest.player_stats import normalize_player_stats
from src.ingest.run import persist_tables
from src.utils.io import read_parquet

# ---- Metric columns to extract from the FBref-style wide CSVs ----
# (column_name_in_csv, canonical_metric_name)
METRICS = [
    # Playing time
    ("MP", "apps"),
    ("Starts", "starts"),
    ("Min", "minutes"),
    ("90s", "nineties"),
    ("Mn/MP", "minutes_per_app"),
    ("Min%", "minutes_pct"),
    ("Mn/Start", "minutes_per_start"),
    ("Compl", "complete_matches"),
    ("Subs", "sub_appearances"),
    ("Mn/Sub", "minutes_per_sub"),
    ("unSub", "unused_sub"),
    ("PPM", "points_per_match"),
    # Goals & assists
    ("Gls", "goals"),
    ("Ast", "assists"),
    ("G+A", "goal_assist"),
    ("G-PK", "non_penalty_goals"),
    ("PK", "penalties_scored"),
    ("PKatt", "penalties_attempted"),
    # Expected
    ("xG", "xg"),
    ("npxG", "npxg"),
    ("xAG", "xag"),
    ("npxG+xAG", "npxg_xag"),
    # Progression
    ("PrgC", "progressive_carries"),
    ("PrgP", "progressive_passes"),
    ("PrgR", "progressive_receives"),
    # Shooting
    ("Sh", "shots"),
    ("SoT", "shots_on_target"),
    ("SoT%", "shot_accuracy"),
    ("Sh/90", "shots_per90"),
    ("SoT/90", "shots_on_target_per90"),
    ("G/Sh", "goal_per_shot"),
    ("G/SoT", "goal_per_shot_on_target"),
    # Defensive / misc
    ("Fls", "fouls"),
    ("Fld", "fouled"),
    ("Off", "offsides"),
    ("Crs", "crosses"),
    ("Int", "interceptions"),
    ("TklW", "tackles_won"),
    ("OG", "own_goals"),
    # Cards
    ("CrdY", "yellow_cards"),
    ("CrdR", "red_cards"),
    ("2CrdY", "double_yellows"),
    # Goalkeeper
    ("GA", "goals_against"),
    ("GA90", "goals_against_per90"),
    ("SoTA", "shots_on_target_against"),
    ("Saves", "saves"),
    ("Save%", "save_pct"),
    ("CS", "clean_sheets"),
    ("CS%", "clean_sheet_pct"),
    ("W", "keeper_wins"),
    ("D", "keeper_draws"),
    ("L", "keeper_losses"),
    ("PKA", "penalties_against"),
    ("PKsv", "penalties_saved"),
    ("PKm", "penalties_missed"),
    # On-field impact
    ("onG", "on_goals_for"),
    ("onGA", "on_goals_against"),
    ("+/-", "plus_minus"),
    ("+/-90", "plus_minus_per90"),
    ("On-Off", "on_off_diff"),
    # Non-penalty combined
    ("G+A-PK", "goal_assist_non_penalty"),
]


def load_and_normalize(csv_path, season, source="fbref"):
    """Load a Kaggle FBref-style CSV, extract known metrics, return canonical long frame."""
    raw = pd.read_csv(csv_path)

    # Standardize id columns
    id_map = {"Player": "player", "Squad": "team", "Pos": "position"}
    df = raw.rename(columns=id_map)[["player", "team", "position"]].copy()

    # Extract each metric that exists in this CSV
    long_frames = []
    for csv_col, metric_name in METRICS:
        if csv_col in raw.columns:
            sub = df.copy()
            sub["metric"] = metric_name
            # Some columns may be empty strings, coerce to float, NaN stays NaN
            sub["value"] = pd.to_numeric(raw[csv_col], errors="coerce")
            long_frames.append(sub.dropna(subset=["value"]))

    if not long_frames:
        return pd.DataFrame(columns=["player", "team", "season", "position", "metric", "value", "source"])

    result = pd.concat(long_frames, ignore_index=True)
    result["season"] = str(season)
    result["source"] = source
    return result


def main():
    # Only datasets with matching column format (FBref-style wide)
    files = [
        ("data/raw/kaggle/players_data-2025_2026.csv", "2025-2026"),
        ("data/raw/kaggle/players_data-2024_2025.csv", "2024-2025"),
    ]

    all_frames = []
    for path, season in files:
        try:
            df = load_and_normalize(path, season)
            print(f"{path}: {len(df):,} rows (players: {df['player'].nunique()}, teams: {df['team'].nunique()})")
            all_frames.append(df)
        except Exception as e:
            import traceback
            print(f"{path}: ERROR: {e}")
            traceback.print_exc()

    combined = pd.concat(all_frames, ignore_index=True)

    # Deduplicate: same player/team/season/position/metric keeps max value
    # (some CSVs have duplicate rows for multi-position players)
    gcols = ["player", "team", "season", "position", "metric", "source"]
    combined = combined.groupby(gcols, as_index=False)["value"].max()

    # Load existing player_stats (placeholders) and merge
    existing = read_parquet("data/processed/player_stats.parquet")
    # Drop placeholder rows (single-row test data)
    existing_real = existing[existing["source"] != "fbref"] if "source" in existing.columns else existing.iloc[0:0]

    final = pd.concat([combined, existing_real], ignore_index=True)

    print(f"\nTotal player_stats: {len(final):,} rows")
    print(f"  Players: {final['player'].nunique():,}")
    print(f"  Teams: {final['team'].nunique():,}")
    print(f"  Seasons: {sorted(final['season'].unique())}")
    print(f"  Metrics: {sorted(final['metric'].unique())}")
    print(f"  Sources: {final['source'].value_counts().to_dict()}")

    persist_tables({"player_stats": final}, out_dir="data/processed")
    print("Written to data/processed/player_stats.parquet")


if __name__ == "__main__":
    main()
