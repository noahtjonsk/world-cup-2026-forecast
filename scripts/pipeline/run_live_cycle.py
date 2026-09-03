"""Run one full in-tournament update against data/processed.

Two steps:

  1. Optionally fetch newly finished results into matches.parquet and mark those
     fixtures played. Skipped when APIFOOTBALL_KEY is not set.
  2. Replay Elo from the tournament cutoff, snapshot features for the fixtures
     still to come, produce W/D/L and corrected expected goals, then rerun the
     full 10,000-run Monte-Carlo. Persists ratings, matchup_features,
     match_predictions, simulation_results and bracket_results, and rewrites
     reports/simulation.md.

Predicted elevens are re-keyed from fixture_id to match_id, which is what the
update loop expects. Takes about four to five minutes. The dashboard picks the new
numbers up on the next page load.

    python scripts/pipeline/run_live_cycle.py              # fetches if the key is set
    python scripts/pipeline/run_live_cycle.py --no-fetch   # matches.parquet updated by hand"""
import argparse
import os

import pandas as pd

from src.config import load_params
from src.simulation.format import load_tournament_format
from src.simulation.montecarlo import champion_odds
from src.states.elo_update import seed_from_ratings
from src.states.runner import run_live_update
from src.utils.io import read_parquet
from src.utils.ids import make_match_id


def main(do_fetch=True):
    if do_fetch:
        if os.environ.get("APIFOOTBALL_KEY"):
            from scripts.ingest.fetch_live_results import main as fetch_main
            fetch_main(dry_run=False)
        else:
            print("APIFOOTBALL_KEY not set - skipping fetch (using matches.parquet as-is)")

    matches = read_parquet("data/processed/matches.parquet")
    fixtures = read_parquet("data/processed/fixtures.parquet")
    rt = read_parquet("data/processed/team_ratings.parquet")
    ps = read_parquet("data/processed/player_stats.parquet")
    ts = read_parquet("data/processed/team_style.parquet")
    lineups = read_parquet("data/processed/lineups.parquet")

    cut = load_params()["states"]["live_cutoff_date"]
    seed = seed_from_ratings(rt, cut)
    fmt = load_tournament_format("2026")

    # re-key predicted XIs to match_id (the snapshot builder's join key)
    fx = fixtures.copy()
    fx["match_id"] = [make_match_id(pd.Timestamp(d), h, a)
                      for d, h, a in zip(fx["date"], fx["home_team"], fx["away_team"])]
    id_map = dict(zip(fx["fixture_id"], fx["match_id"]))
    lu = lineups.copy()
    lu["match_id"] = lu["fixture_id"].map(id_map)
    lu = lu.dropna(subset=["match_id"])

    preds = run_live_update(matches, fixtures, seed, fmt, ps, ts, lineups=lu,
                            base_ratings=rt, out_dir="data/processed")
    print(f"\nlive cycle complete: {len(preds)} remaining fixtures forecast")
    odds = champion_odds(read_parquet("data/processed/simulation_results.parquet"), top=5)
    print(odds[["team", "prob", "ci_low", "ci_high"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    main(do_fetch=not a.no_fetch)
