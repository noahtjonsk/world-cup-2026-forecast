"""Rebuild fixtures.parquet and groups.parquet from the post-draw Wikipedia schedule.

The first version of these tables was generated rather than transcribed, and it was
wrong in two ways that nothing detected: Curacao and Panama were swapped between
Groups E and L, and four host fixtures in Groups B and D had home and away
reversed, which meant home advantage was applied to the wrong side. Both are fixed
here by reading the real schedule instead of constructing one.

Source is a scrape of the Wikipedia 2026 World Cup page at .firecrawl/wc2026.md.
Refresh it with:

    firecrawl scrape "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup" \
        --only-main-content -o .firecrawl/wc2026.md

Every check below is a hard failure, not a warning:

  - exactly 72 group matches, using Wikipedia match numbers 1 to 72 once each
  - 12 groups of 4 teams, each group's 6 fixtures covering all 6 pairs exactly once
  - every team name resolving to the canonical name used elsewhere in the project
  - every fixture dated inside the group stage window, 2026-06-11 to 2026-06-27

    python scripts/ingest/regenerate_fixtures.py"""
import re
import sys

import pandas as pd

from src.schema import CANON_FIXTURE_COLS, CANON_GROUP_COLS
from src.utils.io import write_parquet

SCRAPE = ".firecrawl/wc2026.md"
# wiki display name -> canonical repo name (matches.parquet / team_ratings spelling)
NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "Côte d’Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
}

TEAM_LINK = r"\[([^\]]+)\]\(https://en\.wikipedia\.org/wiki/[^)]*national[^)]*team[^)]*\)"
PAIR_ROW = re.compile(r"\|\s*" + TEAM_LINK + r".*?\[Match (\d+)\].*?" + TEAM_LINK)
DATE_LINE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
STANDING_ROW = re.compile(r"\|\s*[1-4]\s*\|.*?" + TEAM_LINK)


def parse_groups_and_fixtures(text):
    sections = re.split(r"### (Group [A-L])\n", text)
    groups, fixtures = {}, []
    for i in range(1, len(sections), 2):
        gname = sections[i].split()[-1]
        body = sections[i + 1].split("### ")[0]
        # membership in standings-table order
        members = []
        for m in STANDING_ROW.finditer(body):
            t = NAME_MAP.get(m.group(1), m.group(1))
            if t not in members:
                members.append(t)
        groups[gname] = members[:4]
        # fixtures: most recent "(YYYY-MM-DD)" line above each pairing row
        current_date = None
        for line in body.splitlines():
            d = DATE_LINE.search(line)
            if d and "Match" not in line:
                current_date = d.group(1)
            p = PAIR_ROW.search(line)
            if p:
                home, n, away = p.group(1), int(p.group(2)), p.group(3)
                fixtures.append({
                    "match_n": n, "date": current_date, "group": gname,
                    "home_team": NAME_MAP.get(home, home),
                    "away_team": NAME_MAP.get(away, away),
                })
    return groups, fixtures


def validate(groups, fixtures, canonical_teams):
    errs = []
    if len(fixtures) != 72:
        errs.append(f"expected 72 fixtures, parsed {len(fixtures)}")
    nums = sorted(f["match_n"] for f in fixtures)
    if nums != list(range(1, 73)):
        errs.append(f"match numbers not 1..72: missing {set(range(1,73)) - set(nums)}, dupes {[n for n in set(nums) if nums.count(n) > 1]}")
    if len(groups) != 12 or any(len(t) != 4 for t in groups.values()):
        errs.append(f"groups malformed: { {g: len(t) for g, t in groups.items()} }")
    for g, members in groups.items():
        gfx = [f for f in fixtures if f["group"] == g]
        if len(gfx) != 6:
            errs.append(f"group {g}: {len(gfx)} fixtures, expected 6")
        want = {frozenset(p) for p in __import__("itertools").combinations(members, 2)}
        got = [frozenset((f["home_team"], f["away_team"])) for f in gfx]
        if set(got) != want or len(got) != len(set(got)):
            errs.append(f"group {g}: pairings {sorted(map(sorted, got))} != all pairs of {members}")
    all_teams = {t for g in groups.values() for t in g}
    unknown = all_teams - canonical_teams
    if unknown:
        errs.append(f"teams not found in matches/team_ratings: {sorted(unknown)}")
    for f in fixtures:
        if not f["date"] or not ("2026-06-11" <= f["date"] <= "2026-06-27"):
            errs.append(f"match {f['match_n']} bad date {f['date']}")
    return errs


def main():
    text = open(SCRAPE, encoding="utf-8").read()
    groups, fixtures = parse_groups_and_fixtures(text)

    m = pd.read_parquet("data/processed/matches.parquet")
    rt = pd.read_parquet("data/processed/team_ratings.parquet")
    canonical = (set(m["home_team"]) | set(m["away_team"])) & set(rt["team"])
    errs = validate(groups, fixtures, canonical)
    if errs:
        print("VALIDATION FAILED, nothing written:")
        for e in errs:
            print(" -", e)
        sys.exit(1)

    old = pd.read_parquet("data/processed/fixtures.parquet")

    fx = pd.DataFrame([{
        "fixture_id": 900000 + f["match_n"],
        "date": pd.Timestamp(f["date"]),
        "competition": "World Cup", "season": "2026", "stage": f"Group {f['group']}",
        "home_team": f["home_team"], "away_team": f["away_team"],
        "status": "SCHEDULED", "venue": None, "neutral": True, "source": "wikipedia_2026",
    } for f in sorted(fixtures, key=lambda f: f["match_n"])])[CANON_FIXTURE_COLS]

    gr = pd.DataFrame([{"tournament": "2026", "group": g, "team": t}
                       for g in sorted(groups) for t in groups[g]])[CANON_GROUP_COLS]

    # report the diff vs the old (tainted) tables before overwriting
    old_pairs = set(zip(old["home_team"], old["away_team"]))
    new_pairs = set(zip(fx["home_team"], fx["away_team"]))
    print(f"pairings changed: {len(new_pairs - old_pairs)} of 72")
    for h, a in sorted(new_pairs - old_pairs):
        print(f"  + {h} vs {a}")
    for h, a in sorted(old_pairs - new_pairs):
        print(f"  - {h} vs {a}")

    write_parquet(fx, "data/processed/fixtures.parquet")
    write_parquet(gr, "data/processed/groups.parquet")
    print("\nwrote data/processed/fixtures.parquet (72) + groups.parquet (48). Groups:")
    for g in sorted(groups):
        print(f"  {g}: {', '.join(groups[g])}")


if __name__ == "__main__":
    main()
