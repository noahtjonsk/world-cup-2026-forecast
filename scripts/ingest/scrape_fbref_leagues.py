"""Scrape FBref standard stats for leagues outside the big five and normalize them.

Extends player_stats past the top five European leagues, which matters because a
large share of World Cup squads play elsewhere and would otherwise have no
statistical coverage at all."""
import pandas as pd
import numpy as np
import sys, os, re, time, json
sys.path.insert(0, ".")
from src.ingest.run import persist_tables
from src.utils.io import read_parquet, write_parquet
from src.schema import CANON_PLAYER_STAT_COLS

# ---- Leagues to scrape (FBref comp_id, league_name, seasons) ----
LEAGUES = [
    (32, "Primeira-Liga", ["2024-2025", "2025-2026"]),       # Portugal
    (23, "Eredivisie", ["2024-2025", "2025-2026"]),          # Netherlands
    (24, "Serie-A-Brazil", ["2024", "2025"]),                # Brazil (calendar year seasons)
    (31, "Liga-MX", ["2024-2025", "2025-2026"]),             # Mexico
    (26, "Super-Lig", ["2024-2025", "2025-2026"]),           # Turkey
    (37, "Belgian-Pro-League", ["2024-2025", "2025-2026"]),  # Belgium
    (40, "Scottish-Premiership", ["2024-2025", "2025-2026"]),# Scotland
    (22, "Major-League-Soccer", ["2024", "2025"]),           # MLS (calendar year)
    (21, "Primera-Division", ["2024", "2025"]),              # Argentina (calendar year)
    (33, "2-Bundesliga", ["2024-2025", "2025-2026"]),        # Germany 2
    (10, "EFL-Championship", ["2024-2025", "2025-2026"]),    # England 2
    (70, "Saudi-Professional-League", ["2024-2025", "2025-2026"]), # Saudi Arabia
    # Additional leagues to improve squad coverage
    (34, "Czech-First-League", ["2024-2025", "2025-2026"]),        # Czech Republic
    (55, "K-League-1", ["2024", "2025"]),                          # South Korea
    (47, "Egyptian-Premier-League", ["2024-2025", "2025-2026"]),   # Egypt
    (79, "South-African-Premier-Division", ["2024-2025", "2025-2026"]), # South Africa
    (27, "Super-League-Greece", ["2024-2025", "2025-2026"]),       # Greece
    (30, "Russian-Premier-League", ["2024-2025", "2025-2026"]),    # Russia
    (56, "Austrian-Bundesliga", ["2024-2025", "2025-2026"]),       # Austria
    # Wave 3: remaining leagues with World Cup players
    (67, "Persian-Gulf-Pro-League", ["2024-2025", "2025-2026"]),    # Iran
    (54, "J1-League", ["2024", "2025"]),                            # Japan
    (52, "Chinese-Super-League", ["2024", "2025"]),                 # China
    (64, "Hrvatska-NL", ["2024-2025", "2025-2026"]),                # Croatia
    (50, "Danish-Superliga", ["2024-2025", "2025-2026"]),           # Denmark
    (59, "Swiss-Super-League", ["2024-2025", "2025-2026"]),         # Switzerland
    (39, "Ukrainian-Premier-League", ["2024-2025", "2025-2026"]),   # Ukraine
    (105, "A-League-Men", ["2024-2025", "2025-2026"]),              # Australia/NZ
    (29, "Allsvenskan", ["2024", "2025"]),                          # Sweden
    (28, "Eliteserien", ["2024", "2025"]),                          # Norway
    # Fixes: correct names for previously failed leagues
    (79, "South-African-Premiership", ["2024-2025", "2025-2026"]),  # South Africa
    (27, "Super-League-Greece", ["2024-2025", "2025-2026"]),        # Greece
]

# FBref column rename map (from MultiIndex flat columns to canonical metric names)
# Standard stats table columns
COLUMN_MAP = {
    "MP": "apps",
    "Starts": "starts",
    "Min": "minutes",
    "90s": "nineties",
    "Gls": "goals",
    "Ast": "assists",
    "G+A": "goal_assist",
    "G-PK": "non_penalty_goals",
    "PK": "penalties_scored",
    "PKatt": "penalties_attempted",
    "CrdY": "yellow_cards",
    "CrdR": "red_cards",
}
# Per-90 versions (suffix _p90)
PER90_COLS = {"Gls", "Ast", "G+A", "G-PK", "G+A-PK"}


def build_url(comp_id, season, league_name):
    """Build FBref standard stats URL."""
    return f"https://fbref.com/en/comps/{comp_id}/{season}/stats/{season}-{league_name}-Stats"


def scrape_page(url, out_path):
    """Scrape a single FBref page via firecrawl, save raw HTML."""
    cmd = f'firecrawl scrape "{url}" --only-main-content --format rawHtml -o "{out_path}"'
    ret = os.system(cmd)
    return ret == 0


def parse_player_table(html_path):
    """Parse the FBref HTML page, extract the player stats table, return clean DataFrame
    with unique column names (Performance section preferred over Per 90 Minutes duplicates)."""
    try:
        tables = pd.read_html(html_path)
    except Exception as e:
        print(f"    read_html failed: {e}")
        return None

    for t in tables:
        if t.shape[0] > 100 and t.shape[1] > 15:
            df = t.copy()

            if isinstance(df.columns, pd.MultiIndex):
                # MultiIndex like: ('Playing Time', 'MP'), ('Performance', 'Gls'), ('Per 90 Minutes', 'Gls')
                # Strategy: keep 'Playing Time' and 'Performance' section columns; drop 'Per 90 Minutes' duplicates
                new_cols = []
                drop_indices = []
                for i, col in enumerate(df.columns):
                    section = col[0] if isinstance(col, tuple) else ''
                    name = col[-1] if isinstance(col, tuple) else col
                    if 'Per 90' in str(section) or 'Per90' in str(section):
                        drop_indices.append(i)
                        continue
                    new_cols.append(name)
                # Remove duplicate columns (the Per 90 duplicates)
                df = df.iloc[:, [i for i in range(len(df.columns)) if i not in drop_indices]]
                # Now set flattened column names
                df.columns = new_cols
                # If there are still duplicate names (from 'Playing Time' vs 'Performance' overlap), dedupe
                if df.columns.duplicated().any():
                    dupes = df.columns[df.columns.duplicated()].tolist()
                    # Keep first occurrence
                    df = df.loc[:, ~df.columns.duplicated(keep='first')]

            return df
    return None


def normalize_fbref_table(df, season, source="fbref"):
    """Convert a parsed FBref player table to canonical long format."""
    # Rename known columns
    rename = {}
    for col in df.columns:
        if col in COLUMN_MAP:
            rename[col] = COLUMN_MAP[col]
        elif col in PER90_COLS:
            # These are the per-90 duplicates, skip them (we have the raw counts)
            pass

    # Extract id columns
    id_cols_present = [c for c in ["Player", "Nation", "Pos", "Squad"] if c in df.columns]
    if "Player" not in id_cols_present or "Squad" not in id_cols_present:
        print("    Missing Player/Squad columns")
        return None

    out_rows = []
    for _, row in df.iterrows():
        player = str(row["Player"])
        team = str(row["Squad"])
        position = str(row.get("Pos", ""))
        if player in ("Player", "nan", "") or team in ("Squad", "nan", ""):
            continue

        for src_col, metric_name in COLUMN_MAP.items():
            if src_col in df.columns:
                val = row[src_col]
                if pd.isna(val):
                    continue
                try:
                    val = float(val)
                    out_rows.append({
                        "player": player.strip(),
                        "team": team.strip(),
                        "season": str(season),
                        "position": position.strip(),
                        "metric": metric_name,
                        "value": val,
                        "source": source,
                    })
                except (ValueError, TypeError):
                    pass

    if not out_rows:
        return None
    result = pd.DataFrame(out_rows)
    return result[CANON_PLAYER_STAT_COLS]


def main():
    # Load existing player_stats
    existing = read_parquet("data/processed/player_stats.parquet")

    total_added = 0
    for comp_id, league_name, seasons in LEAGUES:
        for season in seasons:
            url = build_url(comp_id, season, league_name)
            safe_name = f"{league_name}-{season}".replace(" ", "-").lower()
            html_path = f".firecrawl/fbref-{safe_name}.html"

            print(f"\n--- {league_name} {season} ---")
            print(f"  URL: {url}")

            if not os.path.exists(html_path) or os.path.getsize(html_path) < 1000:
                print(f"  Scraping...")
                if not scrape_page(url, html_path):
                    print(f"  SCRAPE FAILED, skipping")
                    continue
                time.sleep(2)  # Be polite to firecrawl/FBref

            df = parse_player_table(html_path)
            if df is None:
                print(f"  PARSE FAILED")
                continue

            long = normalize_fbref_table(df, season)
            if long is None or len(long) == 0:
                print(f"  NORMALIZE FAILED")
                continue

            print(f"  Parsed: {len(df)} players -> {len(long):,} metric rows")
            # print sample teams (skip if encoding issues)
            try:
                sample = long['team'].unique()[:5].tolist()
                print(f"  Teams sample: {len(long['team'].unique())} teams, first: {sample[0] if sample else '?'}")
            except:
                pass

            # Append to existing
            existing = pd.concat([existing, long], ignore_index=True)
            total_added += len(long)

    # Deduplicate
    gcols = ["player", "team", "season", "position", "metric", "source"]
    before = len(existing)
    existing = existing.groupby(gcols, as_index=False)["value"].max()
    print(f"\nDedup: {before:,} -> {len(existing):,} rows ({before - len(existing)} removed)")

    print(f"\nTotal added this run: {total_added:,} rows")
    print(f"Total player_stats: {len(existing):,} rows")
    print(f"Players: {existing['player'].nunique():,}")
    print(f"Teams: {existing['team'].nunique():,}")
    print(f"Leagues covered (teams): {sorted(existing['team'].unique())[:30]}...")

    persist_tables({"player_stats": existing}, out_dir="data/processed")
    print("Written to data/processed/player_stats.parquet")


if __name__ == "__main__":
    main()
