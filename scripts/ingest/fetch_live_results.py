"""Fetch finished 2026 World Cup results from API-Football and fold them into the data.

Appends canonical rows to matches.parquet, keyed on match_id so re-running never
double-counts, and marks the matching fixtures as finished so they drop out of the
upcoming list. Every fetched team name is checked against the canonical 48; an
unmapped name fails loudly rather than quietly creating a new team.

Needs an API key in the APIFOOTBALL_KEY environment variable (there is a free
tier at https://www.api-football.com). The World Cup is league id 1, season 2026.

    python scripts/ingest/fetch_live_results.py
    python scripts/ingest/fetch_live_results.py --dry-run   # report, write nothing"""
import argparse
import os
import sys

import pandas as pd

from src.ingest.apifootball import fetch, normalize_results
from src.ingest.results import align_results_to_fixtures, append_new_matches, mark_fixtures_finished
from src.utils.io import read_parquet, write_parquet

LEAGUE_ID = 1          # API-Football: FIFA World Cup
SEASON = 2026


def main(dry_run=False, date_from="2026-06-11", date_to=None):
    key = os.environ.get("APIFOOTBALL_KEY")
    if not key:
        sys.exit("APIFOOTBALL_KEY is not set. Get a key at https://www.api-football.com "
                 "(free tier suffices) and export it, or update matches.parquet manually.")
    date_to = date_to or pd.Timestamp.today().strftime("%Y-%m-%d")

    payload = fetch("fixtures", {"league": LEAGUE_ID, "season": SEASON,
                                 "from": date_from, "to": date_to}, key)
    results = normalize_results(payload)
    print(f"API returned {len(payload.get('response', []))} fixtures, "
          f"{len(results)} finished")
    if results.empty:
        print("nothing to ingest")
        return

    fixtures = read_parquet("data/processed/fixtures.parquet")
    matches = read_parquet("data/processed/matches.parquet")

    # every fetched name must resolve to our canonical vocabulary, fail loudly
    known = set(fixtures["home_team"]) | set(fixtures["away_team"])
    unknown = (set(results["home_team"]) | set(results["away_team"])) - known
    if unknown:
        sys.exit(f"unmapped team names from the API (extend APIFOOTBALL_TEAM_ALIASES): "
                 f"{sorted(unknown)}")

    matched, unmatched = align_results_to_fixtures(results, fixtures)
    if not unmatched.empty:
        print("WARNING - results with no matching fixture (NOT ingested):")
        print(unmatched.to_string(index=False))

    before = len(matches)
    matches2 = append_new_matches(matches, matched)
    fixtures2 = mark_fixtures_finished(fixtures, matched)
    n_new = len(matches2) - before
    n_ft = int((fixtures2["status"] == "FT").sum())
    print(f"{n_new} new result(s); {n_ft} fixtures now FT")
    print(matched[["date", "home_team", "home_score", "away_score", "away_team"]]
          .to_string(index=False))

    if dry_run:
        print("--dry-run: nothing written")
        return
    write_parquet(matches2, "data/processed/matches.parquet")
    write_parquet(fixtures2, "data/processed/fixtures.parquet")
    print("wrote matches.parquet + fixtures.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--from", dest="date_from", default="2026-06-11")
    ap.add_argument("--to", dest="date_to", default=None)
    a = ap.parse_args()
    main(dry_run=a.dry_run, date_from=a.date_from, date_to=a.date_to)
