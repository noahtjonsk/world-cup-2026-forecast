"""Report how much of each 2026 squad the player statistics actually cover.

Parses the Wikipedia squads page, matches every named player against
player_stats, and prints the coverage per nation. Low coverage for a nation means
its squad-strength score rests on few players and is shrunk accordingly."""
import pandas as pd
import re, sys
sys.path.insert(0, ".")
from src.utils.io import read_parquet
from src.utils.ids import normalize_name

SQUADS_PATH = ".firecrawl/wikipedia-squads.md"

def extract_club_name(cell):
    """Extract clean club name from Wikipedia table cell.
    Cells look like: [![flag](img)](assoc) [Club Name](club_url)
    We want the display text of the last markdown link."""
    # Remove flag icons: [![...](...)](...) patterns
    cleaned = re.sub(r'\[!\[.*?\]\(.*?\)\]\(.*?\)', '', cell)
    # Extract text from markdown links: [text](url) or [text](url "title")
    links = re.findall(r'\[([^\]]+)\]\([^\)]+\)', cleaned)
    if links:
        return links[-1].strip()
    # Fallback: just strip remaining wiki markup
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    cleaned = re.sub(r'\(.*?\)', '', cleaned)
    return cleaned.strip()


def parse_squads(path):
    """Parse Wikipedia 2026 World Cup squads page into list of dicts."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Find all team sections: ### TeamName followed by a table
    # Pattern: ### TeamName \n\n ... \n| No. | Pos. | Player | ... |
    teams = re.split(r"\n### ", text)[1:]  # Split on ### Team headers

    players = []
    for section in teams:
        lines = section.split("\n")
        team_name = lines[0].strip().replace("[", "").replace("]", "")
        if not team_name or "Group" in team_name:
            continue

        # Find the table rows
        in_table = False
        for line in lines:
            if line.startswith("|") and "No." in line and "Pos." in line:
                in_table = True
                continue
            if in_table and line.startswith("|") and "---" not in line:
                cols = [c.strip() for c in line.split("|")[1:-1]]
                if len(cols) >= 7:
                    # No. | Pos. | Player | DOB(age) | Caps | Goals | Club
                    try:
                        number = cols[0]
                        position = cols[1].split("[")[0].strip()  # Remove wiki links
                        # Extract player name: strip italic/wiki markup and captain tags
                        player_cell = cols[2]
                        # Remove italic markup like _( [captain](url) )_ or _([text])_
                        player_cell = re.sub(r'_\(\s*\[captain\]\([^)]+\)\s*\)_', '', player_cell)
                        # Remove remaining italic markup
                        player_cell = re.sub(r'_\([^)]*\)_', '', player_cell)
                        # Extract name from wiki link [Name](url)
                        name_match = re.search(r'\[([^\]]+)\]\(https://en\.wikipedia', player_cell)
                        if name_match:
                            player_name = name_match.group(1).strip()
                        else:
                            # Fallback: get text between first [ and ]
                            player_name = player_cell.split('[')[-1].split(']')[0].strip()
                            if not player_name or '[' in player_name:
                                player_name = player_cell.strip()
                        club = extract_club_name(cols[6])
                        # Extract caps and goals (plain numbers in cols[4] and [5])
                        caps = re.sub(r'[^0-9]', '', cols[4]) if len(cols) > 4 else '0'
                        goals = re.sub(r'[^0-9]', '', cols[5]) if len(cols) > 5 else '0'

                        players.append({
                            "nation": team_name,
                            "player": player_name,
                            "position": position,
                            "club": club,
                            "number": number,
                            "caps": int(caps) if caps else 0,
                            "goals": int(goals) if goals else 0,
                        })
                    except (IndexError, ValueError):
                        pass

    return pd.DataFrame(players)


def main():
    print("Parsing Wikipedia squads...")
    squads = parse_squads(SQUADS_PATH)
    print(f"  Found {len(squads)} players across {squads['nation'].nunique()} nations")

    # Show sample
    print(f"\nSample squads:")
    for nat in list(squads["nation"].unique())[:5]:
        sub = squads[squads["nation"] == nat]
        print(f"  {nat}: {len(sub)} players")
        # Skip detailed print to avoid encoding issues

    # Load player_stats
    ps = read_parquet("data/processed/player_stats.parquet")
    ps_players = set(ps["player"].apply(normalize_name).unique())

    # Cross-reference, try both original name and ASCII-normalized version
    import unicodedata
    def ascii_normalize(name):
        """Strip diacritics: Štěpán -> Stepan"""
        return unicodedata.normalize('NFKD', str(name)).encode('ascii', 'ignore').decode()

    squads["name_normalized"] = squads["player"].apply(normalize_name)
    squads["name_ascii"] = squads["player"].apply(ascii_normalize).apply(normalize_name)
    squads["in_stats"] = squads["name_normalized"].isin(ps_players) | squads["name_ascii"].isin(ps_players)

    # Safety: filter any remaining wiki artifacts that weren't properly parsed
    squads = squads[~squads["player"].str.lower().isin(['captain', ''])].copy()

    # Report
    covered = squads[squads["in_stats"]]
    uncovered = squads[~squads["in_stats"]]

    print(f"\n=== COVERAGE ===")
    print(f"Total squad players: {len(squads)}")
    print(f"Covered in player_stats: {len(covered)} ({100*len(covered)/len(squads):.1f}%)")
    print(f"Missing: {len(uncovered)} ({100*len(uncovered)/len(squads):.1f}%)")

    # Per nation coverage, write to file to avoid encoding issues
    nation_coverage = squads.groupby("nation").agg(
        total=("player", "count"),
        covered=("in_stats", "sum"),
    )
    nation_coverage["pct"] = 100 * nation_coverage["covered"] / nation_coverage["total"]
    nation_coverage = nation_coverage.sort_values("pct")

    with open("data/processed/coverage_report.txt", "w", encoding="utf-8") as f:
        f.write(f"=== WORLD CUP 2026 PLAYER COVERAGE ===\n")
        f.write(f"Total squad players: {len(squads)}\n")
        f.write(f"Covered in player_stats: {len(covered)} ({100*len(covered)/len(squads):.1f}%)\n")
        f.write(f"Missing: {len(uncovered)} ({100*len(uncovered)/len(squads):.1f}%)\n\n")
        f.write(f"Coverage by nation:\n")
        for nat, row in nation_coverage.iterrows():
            bar = "#" * int(row["pct"] / 5) + "-" * (20 - int(row["pct"] / 5))
            f.write(f"  {nat:30s} {bar} {row['covered']:3.0f}/{row['total']:3.0f} ({row['pct']:.0f}%)\n")
        f.write(f"\n=== MISSING PLAYERS (first 50) ===\n")
        for _, p in uncovered.head(50).iterrows():
            f.write(f"  {p['player']:30s} | {p['nation']:20s} | {p['club']}\n")

    print(f"\nCoverage report saved to data/processed/coverage_report.txt")

    # Summary by confederation
    confed_map = {
        "Canada": "CONCACAF", "Mexico": "CONCACAF", "United States": "CONCACAF",
        "Curaçao": "CONCACAF", "Haiti": "CONCACAF", "Panama": "CONCACAF",
        "Argentina": "CONMEBOL", "Brazil": "CONMEBOL", "Colombia": "CONMEBOL",
        "Ecuador": "CONMEBOL", "Paraguay": "CONMEBOL", "Uruguay": "CONMEBOL",
        "England": "UEFA", "France": "UEFA", "Germany": "UEFA", "Spain": "UEFA",
        "Portugal": "UEFA", "Netherlands": "UEFA", "Belgium": "UEFA", "Croatia": "UEFA",
        "Switzerland": "UEFA", "Austria": "UEFA", "Scotland": "UEFA", "Sweden": "UEFA",
        "Norway": "UEFA", "Turkey": "UEFA", "Czech Republic": "UEFA",
        "Bosnia and Herzegovina": "UEFA",
        "Japan": "AFC", "South Korea": "AFC", "Australia": "AFC", "Iran": "AFC",
        "Iraq": "AFC", "Jordan": "AFC", "Qatar": "AFC", "Saudi Arabia": "AFC",
        "Uzbekistan": "AFC",
        "Morocco": "CAF", "Senegal": "CAF", "Egypt": "CAF", "Algeria": "CAF",
        "Ghana": "CAF", "Ivory Coast": "CAF", "Tunisia": "CAF", "South Africa": "CAF",
        "Cape Verde": "CAF", "DR Congo": "CAF",
        "New Zealand": "OFC",
    }

    # Save for later use
    squads.to_csv("data/processed/squad_coverage.csv", index=False)
    print(f"\nSaved to data/processed/squad_coverage.csv")


if __name__ == "__main__":
    main()
