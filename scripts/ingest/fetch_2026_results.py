"""Fetch the real 2026 World Cup results from Wikipedia into results_2026.parquet.

Writes its own table rather than appending to matches.parquet. The published forecast
is only worth anything because it predates the tournament, so the training data must
stay free of the results it was predicting.

Two pages are scraped, because there is no separate group-stage article:

    https://en.wikipedia.org/wiki/2026_FIFA_World_Cup                  72 group matches
    https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage   32 knockout matches

Every validation is a hard failure. The strongest of them recomputes each group table
from the parsed results and compares it against the standings table published on the
same page, which catches a misread scoreline that counts alone would miss.

    python scripts/ingest/fetch_2026_results.py --dry-run   # parse and check, write nothing
    python scripts/ingest/fetch_2026_results.py             # write the table
    python scripts/ingest/fetch_2026_results.py --no-scrape # reuse the cached markdown
"""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")

from src.ingest.wikipedia_results import (          # noqa: E402
    parse_group_results, parse_group_standings, parse_knockout_results, to_frame, validate,
)
from src.utils.io import read_parquet, write_parquet  # noqa: E402

PAGES = {
    ".firecrawl/wc2026-final.md": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup",
    ".firecrawl/wc2026-knockout.md": "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_knockout_stage",
}
OUT = "data/processed/results_2026.parquet"


def scrape(path, url):
    print(f"  scraping {url}")
    cmd = ["firecrawl", "scrape", url, "--only-main-content", "-o", path]
    subprocess.run(cmd, check=True, shell=False)


def main(argv):
    dry_run = "--dry-run" in argv
    no_scrape = "--no-scrape" in argv

    for path, url in PAGES.items():
        if no_scrape or Path(path).exists():
            print(f"  using cached {path}")
        else:
            scrape(path, url)
        if not Path(path).exists():
            sys.exit(f"missing {path}; drop --no-scrape to fetch it")

    main_md = Path(".firecrawl/wc2026-final.md").read_text(encoding="utf-8")
    ko_md = Path(".firecrawl/wc2026-knockout.md").read_text(encoding="utf-8")

    groups = parse_group_results(main_md)
    knockouts = parse_knockout_results(ko_md)
    published = parse_group_standings(main_md)
    print(f"\nparsed {len(groups)} group and {len(knockouts)} knockout matches")

    canonical = set(read_parquet("data/processed/groups.parquet")["team"])
    errs = validate(groups, knockouts, published, canonical)
    if errs:
        print(f"\nVALIDATION FAILED, nothing written ({len(errs)} problems):")
        for e in errs[:25]:
            print("  -", e)
        sys.exit(1)
    print("validation passed: counts, dates, canonical names, and every group table "
          "reproduced from the results")

    df = to_frame(groups, knockouts)

    # The predictions must line up one-for-one with the group results, or the scorecard
    # would silently score a subset.
    preds = read_parquet("data/processed/match_predictions.parquet")
    ids = set(df["match_id"])
    missing = set(preds["match_id"]) - ids
    if missing:
        print(f"\n{len(missing)} of {len(preds)} predictions have no matching result:")
        for mid in list(missing)[:10]:
            row = preds[preds["match_id"] == mid].iloc[0]
            print(f"  - {row['date'].date()} {row['home_team']} vs {row['away_team']}")
        sys.exit(1)
    print(f"all {len(preds)} predictions join to a result")

    if dry_run:
        print("\n--dry-run: nothing written")
        return
    write_parquet(df, OUT)
    print(f"\nwrote {len(df)} results to {OUT}")


if __name__ == "__main__":
    main(sys.argv[1:])
