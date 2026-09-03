"""Pull StatsBomb open data and merge it into the canonical matches table.

Merges rather than replaces. An earlier version wrote only the StatsBomb rows, which
silently truncated matches.parquet from 49,555 rows to 334 for anyone who ran it, and
because the module had no entry-point guard, merely importing it was enough to do that.
Both are fixed here: the work happens under `main()`, and existing non-StatsBomb rows
are preserved.

    python scripts/ingest/ingest_statsbomb.py             # merge into matches.parquet
    python scripts/ingest/ingest_statsbomb.py --dry-run   # fetch and report, write nothing
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, ".")

from src.ingest.run import persist_tables            # noqa: E402
from src.ingest.statsbomb import normalize_matches   # noqa: E402
from src.schema import CANON_MATCH_COLS              # noqa: E402
from src.utils.io import read_parquet                # noqa: E402

MATCHES = "data/processed/matches.parquet"

# StatsBomb open data competitions, free and needing no credentials.
COMPETITIONS = [
    # (competition_id, season_id, label)
    (43, 3, "World Cup 2018"),
    (43, 106, "World Cup 2022"),
    (55, 43, "Euros 2020"),
    (11, 90, "Euros 2024"),
    (44, 282, "Copa America 2024"),
    (11, 42, "Euros 2016"),
    (11, 1, "Euros 2012"),
    (55, 282, "Euros 2024"),
]


def fetch():
    """Every configured competition, normalized and concatenated."""
    from statsbombpy import sb

    frames = []
    for comp_id, season_id, label in COMPETITIONS:
        try:
            matches = normalize_matches(sb.matches(competition_id=comp_id,
                                                   season_id=season_id))
            matches["competition"] = label
            frames.append(matches)
            print(f"  {label}: {len(matches)} matches")
        except Exception as e:
            print(f"  {label}: SKIPPED ({e})")
    if not frames:
        return pd.DataFrame(columns=CANON_MATCH_COLS)
    out = pd.concat(frames, ignore_index=True)
    out["source"] = "statsbomb"
    return out


def merge_into_existing(fetched, existing):
    """Replace the StatsBomb rows in `existing` with `fetched`, keeping everything else.

    Keying on source rather than on match_id, because this script owns exactly the rows
    it produced and must not touch rows any other ingest wrote.
    """
    if existing is None or existing.empty:
        return fetched[CANON_MATCH_COLS]
    kept = existing[existing["source"] != "statsbomb"]
    return pd.concat([kept, fetched], ignore_index=True)[CANON_MATCH_COLS]


def main(argv=()):
    dry_run = "--dry-run" in argv
    fetched = fetch()
    if fetched.empty:
        sys.exit("no data pulled; nothing to merge")
    print(f"\nfetched {len(fetched)} matches from StatsBomb")

    existing = read_parquet(MATCHES) if Path(MATCHES).exists() else None
    merged = merge_into_existing(fetched, existing)
    before = 0 if existing is None else len(existing)
    print(f"matches.parquet: {before} rows -> {len(merged)} rows")
    print(f"  sources: {merged['source'].value_counts().to_dict()}")

    if before and len(merged) < before * 0.9:
        sys.exit(f"refusing to write: this would drop {before - len(merged)} rows")

    if dry_run:
        print("\n--dry-run: nothing written")
        return
    persist_tables({"matches": merged}, out_dir="data/processed")
    print(f"written to {MATCHES}")


if __name__ == "__main__":
    main(sys.argv[1:])
