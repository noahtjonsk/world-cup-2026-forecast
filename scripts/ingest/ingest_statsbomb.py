"""Pull StatsBomb open data and append it to the canonical matches table."""
import pandas as pd
import sys
sys.path.insert(0, ".")

from statsbombpy import sb
from src.ingest.statsbomb import normalize_matches
from src.ingest.run import persist_tables

# StatsBomb open data competitions (free, no auth needed)
COMPETITIONS = [
    # (competition_id, season_id, name)
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
    (55, 43, "Euros 2020"),
    (11, 90, "Euros 2024"),
    (44, 282, "Copa America 2024"),
    (11, 42, "Euros 2016"),
    (11, 1, "Euros 2012"),
    (55, 282, "Euros 2024"),  # duplicate but fine
]

all_matches = []
for comp_id, season_id, label in COMPETITIONS:
    try:
        raw = sb.matches(competition_id=comp_id, season_id=season_id)
        matches = normalize_matches(raw)
        matches["competition"] = label
        all_matches.append(matches)
        print(f"  {label}: {len(matches)} matches")
    except Exception as e:
        print(f"  {label}: SKIPPED ({e})")

if all_matches:
    combined = pd.concat(all_matches, ignore_index=True)
    combined["source"] = "statsbomb"
    print(f"\nTotal: {len(combined)} matches from StatsBomb")
    print(f"Teams: {combined['home_team'].nunique()}")
    print(f"Date range: {combined['date'].min()} to {combined['date'].max()}")

    persist_tables({"matches": combined}, out_dir="data/processed")
    print("Written to data/processed/matches.parquet")
else:
    print("No data pulled!")
