"""Replay the 2022 World Cup through the simulator as a sanity check.

Fits the goal model on matches up to that tournament, runs the 32-team bracket,
and prints the champion odds it would have produced. The point is to confirm the
engine gives a plausible field on a tournament whose result we already know."""
import numpy as np
import pandas as pd
from src.utils.io import read_parquet
from src.simulation.format import load_tournament_format
from src.simulation.backtest import backtest_tournament
from src.simulation.montecarlo import champion_odds

def main():
    m = read_parquet("data/processed/matches.parquet")
    fmt = load_tournament_format("2022")
    groups = {
        "A": ["Qatar", "Ecuador", "Senegal", "Netherlands"],
        "B": ["England", "Iran", "USA", "Wales"],
        "C": ["Argentina", "Saudi Arabia", "Mexico", "Poland"],
        "D": ["France", "Australia", "Denmark", "Tunisia"],
        "E": ["Spain", "Costa Rica", "Germany", "Japan"],
        "F": ["Belgium", "Canada", "Morocco", "Croatia"],
        "G": ["Brazil", "Serbia", "Switzerland", "Cameroon"],
        "H": ["Portugal", "Ghana", "Uruguay", "South Korea"],
    }

    print("Fitting Dixon-Coles on pre-tournament matches + 5000 MC runs...")
    out = backtest_tournament(
        m, fmt, groups, tournament_start="2022-11-20",
        tournament="2022", n_runs=5000,
    )

    print("\n=== Title odds (top 8) ===")
    print(champion_odds(out, top=8).to_string(index=False))

    print("\n=== R16 reach probabilities ===")
    r16 = out[out["round"] == "R16"].sort_values("prob", ascending=False)
    print(r16.head(16).to_string(index=False))

    # Actual 2022 R16 teams for comparison
    actual_r16 = {
        "Netherlands", "Senegal", "England", "USA",
        "Argentina", "Poland", "France", "Australia",
        "Japan", "Spain", "Morocco", "Croatia",
        "Brazil", "Switzerland", "Portugal", "South Korea",
    }
    predicted_r16 = set(r16.head(16)["team"])
    print(f"\n=== Comparison ===")
    print(f"Correctly predicted: {len(actual_r16 & predicted_r16)}/16")
    print(f"Missed: {actual_r16 - predicted_r16}")
    print(f"Surprise picks: {predicted_r16 - actual_r16}")

if __name__ == "__main__":
    main()
