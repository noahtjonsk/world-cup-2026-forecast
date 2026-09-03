"""Refit the goal model and rerun the tournament, rewriting reports/simulation.md.

Uses the deployed configuration: competition-tier weighting plus the Elo-anchored
shrinkage prior at the locked strength. The prior is built as of the tournament
cutoff date rather than from the full history, so it never sees results the model
is meant to predict, matching what the live update does.

Lambda, beta and the competition weights all come from config/params.yaml, so
settle those before running. Takes about three minutes for the 10,000-run
Monte-Carlo.

    python scripts/pipeline/run_simulation.py"""
import pandas as pd

from src.config import load_params
from src.simulation.format import load_tournament_format
from src.simulation.run import run_simulation_pipeline
from src.simulation.montecarlo import champion_odds
from src.models.elo_prior import elo_prior_net_asof


def main():
    cfg = load_params()
    dc = cfg["models"]["dixon_coles"]
    cutoff = cfg["states"]["live_cutoff_date"]
    beta = dc.get("prior_scale", 0.35)
    print(f"config: prior_strength(lambda)={dc.get('prior_strength')} | prior_scale(beta)={beta} "
          f"| cutoff={cutoff} | weighting={'ON' if dc.get('competition_weights') else 'OFF'}")

    matches = pd.read_parquet("data/processed/matches.parquet")
    fixtures = pd.read_parquet("data/processed/fixtures.parquet")
    ratings = pd.read_parquet("data/processed/team_ratings.parquet")
    fmt = load_tournament_format("2026")

    fit_teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    pn = elo_prior_net_asof(ratings, fit_teams, cutoff, beta=beta)
    elo_prior = dict(zip(fit_teams, pn))

    out = run_simulation_pipeline(matches, fixtures, fmt, elo_prior=elo_prior,
                                  tournament="2026", report_path="reports/simulation.md")
    print("\n=== title odds (top 12) ===")
    print(champion_odds(out, top=12).to_string(index=False))


if __name__ == "__main__":
    main()
