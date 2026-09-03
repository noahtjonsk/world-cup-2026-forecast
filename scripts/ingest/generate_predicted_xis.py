"""Build a predicted starting eleven for each of the 48 teams.

Takes the Wikipedia squad lists, ranks each squad's players by their quality score
from player_stats, and picks an eleven under role quotas. Idempotent: re-running
replaces the existing lineups table rather than appending to it.

    python scripts/ingest/generate_predicted_xis.py"""
import sys
sys.path.insert(0, ".")
import pandas as pd
from src.ingest.run import persist_tables
from src.schema import CANON_LINEUP_COLS
from src.utils.io import read_parquet
from src.utils.ids import normalize_name

# Standard formation: 4-3-3
# Picks: 1 GK, 2 CB, 1 LB, 1 RB, 3 CM, 2 W, 1 ST
POSITION_ORDER = ["GK", "CB", "CB", "LB", "RB", "CM", "CM", "CM", "W", "W", "ST"]

# Map Wikipedia/FBref positions to our formation slots
POSITION_TO_SLOT = {
    "GK": "GK", "G": "GK",
    "CB": "CB", "DF": "CB", "D": "CB", "DEF": "CB",
    "RCB": "CB", "LCB": "CB",
    "LB": "LB", "LWB": "LB",
    "RB": "RB", "RWB": "RB",
    "DM": "CM", "CDM": "CM",
    "CM": "CM", "LCM": "CM", "RCM": "CM", "MF": "CM", "M": "CM", "MID": "CM",
    "AM": "CM", "CAM": "CM",
    "RM": "W", "LM": "W", "RW": "W", "LW": "W", "W": "W",
    "CF": "ST", "ST": "ST", "SS": "ST", "FW": "ST", "F": "ST", "S": "ST", "FWD": "ST",
}


# Numeric position codes from Wikipedia: 1=GK, 2=DF, 3=MF, 4=FW
NUMERIC_POSITION_MAP = {"1": "GK", "2": "DF", "3": "MF", "4": "FW"}

def slot_position(pos_str):
    """Map a raw position string to one of our formation slots."""
    if pd.isna(pos_str) or not pos_str:
        return "CM"
    code = str(pos_str).strip()
    # Handle numeric codes first (Wikipedia format)
    if code.isdigit():
        code = NUMERIC_POSITION_MAP.get(code, "CM")
    else:
        code = code.upper().split(",")[0].split("/")[0].split()[0]
    return POSITION_TO_SLOT.get(code, "CM")


def build_predicted_xis(squads_csv, player_stats):
    """For each nation, pick best XI based on player_stats quality scores.
    Falls back to most-capped players for low-coverage teams."""
    all_squads = pd.read_csv(squads_csv)  # full squad, including uncovered players
    squads = all_squads[all_squads["in_stats"] == True].copy()
    squads["name_normalized"] = squads["player"].apply(normalize_name)

    # Compute quality score per player from player_stats
    ps = player_stats.copy()
    ps["name_normalized"] = ps["player"].apply(normalize_name)

    # Simple quality: mean of all metric values per player (higher = better)
    quality = ps.groupby("name_normalized")["value"].mean().reset_index()
    quality.columns = ["name_normalized", "quality_score"]

    # Merge with squads
    squads = squads.merge(quality, on="name_normalized", how="left")
    squads["slot"] = squads["position"].apply(slot_position)

    rows = []
    fid = 700000
    for nation in sorted(all_squads["nation"].unique()):
        team_squad = squads[squads["nation"] == nation].copy()

        # If low coverage, use caps-based fallback from FULL Wikipedia squad
        if len(team_squad) < 11:
            print(f"  {nation}: only {len(team_squad)} covered players, using caps fallback")
            full_team = all_squads[all_squads["nation"] == nation].copy()
            full_team["caps"] = pd.to_numeric(full_team.get("caps", 0), errors="coerce").fillna(0)
            team_squad = full_team.nlargest(11, "caps").copy()
            team_squad["name_normalized"] = team_squad["player"].apply(normalize_name)
            team_squad["slot"] = team_squad["position"].apply(slot_position)
            team_squad["quality_score"] = team_squad["caps"]  # use caps as proxy quality
            team_squad["in_stats"] = False

        # Ensure at least one GK: if quality pool has none, take most-capped GK from full squad
        full_team_all = all_squads[all_squads['nation'] == nation].copy()
        full_team_all['caps'] = pd.to_numeric(full_team_all.get('caps', 0), errors='coerce').fillna(0)
        gk_pool = team_squad[team_squad['slot'] == 'GK']
        if len(gk_pool) == 0:
            full_gks = full_team_all[full_team_all['position'].astype(str).str.strip().isin(['1', 'GK'])]
            if len(full_gks):
                best_gk = full_gks.nlargest(1, 'caps').iloc[0]
                best_gk_row = pd.DataFrame([{
                    'player': best_gk['player'], 'position': best_gk['position'],
                    'name_normalized': normalize_name(best_gk['player']),
                    'slot': 'GK', 'quality_score': best_gk['caps'],
                    'nation': nation, 'in_stats': False, 'club': best_gk.get('club','')
                }])
                team_squad = pd.concat([team_squad, best_gk_row], ignore_index=True)

        # For each formation slot, pick the best available player
        picked = set()
        xi = []
        for slot in POSITION_ORDER:
            candidates = team_squad[
                (team_squad["slot"] == slot) &
                (~team_squad["name_normalized"].isin(picked))
            ]
            if len(candidates) == 0:
                candidates = team_squad[~team_squad["name_normalized"].isin(picked)]
            if len(candidates) == 0:
                break
            best = candidates.nlargest(1, "quality_score").iloc[0]
            picked.add(best["name_normalized"])
            xi.append(best)

        if len(xi) < 11:
            print(f"  {nation}: only {len(xi)} players total, skipping")
            continue

        # Get the fixture_ids for this nation's group matches
        fixtures = read_parquet("data/processed/fixtures.parquet")
        nation_fixtures = fixtures[
            (fixtures["home_team"] == nation) | (fixtures["away_team"] == nation)
        ]

        for _, fx in nation_fixtures.iterrows():
            for _, p in enumerate(xi):
                rows.append({
                    "fixture_id": fx["fixture_id"],
                    "team": nation,
                    "player": p["player"],
                    "position": p["position"],
                    "is_starter": True,
                    "formation": "4-3-3",
                    "source": "predicted_xi",
                })
            # Add bench: remaining squad players not in XI
            bench = team_squad[~team_squad["name_normalized"].isin(picked)]
            for _, p in bench.iterrows():
                rows.append({
                    "fixture_id": fx["fixture_id"],
                    "team": nation,
                    "player": p["player"],
                    "position": p["position"],
                    "is_starter": False,
                    "formation": "4-3-3",
                    "source": "predicted_xi",
                })

    df = pd.DataFrame(rows)[CANON_LINEUP_COLS]
    return df


def main():
    print("Building predicted 2026 XIs...")
    ps = read_parquet("data/processed/player_stats.parquet")
    squads_csv = "data/processed/squad_coverage.csv"

    df = build_predicted_xis(squads_csv, ps)
    print(f"\nPredicted XIs: {len(df):,} rows")
    print(f"Fixtures covered: {df['fixture_id'].nunique()}")
    print(f"Teams: {df['team'].nunique()}")
    print(f"Players: {df['player'].nunique()}")

    # Merge with existing lineups (replace placeholders)
    existing = read_parquet("data/processed/lineups.parquet")
    # Remove old placeholder rows
    existing_real = existing[~existing["source"].isin(["apifootball", "predicted_xi"])] if len(existing) > 2 else existing.iloc[0:0]
    final = pd.concat([df, existing_real], ignore_index=True)

    persist_tables({"lineups": final}, out_dir="data/processed")
    print("Saved to data/processed/lineups.parquet")


if __name__ == "__main__":
    main()
