# src/ingest/wikipedia_results.py
"""Parse 2026 World Cup results out of scraped Wikipedia markdown.

Pure text in, rows out. The scraping itself lives in
scripts/ingest/fetch_2026_results.py, so everything here is testable against a
fixture string without a network.

Two quirks of the source drive the parsing. Team cells carry an inline flag image
between the link and the cell delimiter, so images are stripped before anything else
is matched. And the score is a link on the main article and in the round of 32
(`[2-0](url)`) but plain text from the round of 16 onward (`0-3`), so the brackets
have to be optional.
"""
import re

import pandas as pd

from src.schema import CANON_RESULT_COLS
from src.utils.ids import make_match_id

# Wikipedia display name -> the canonical name used in matches.parquet / team_ratings.
NAME_MAP = {
    "Côte d'Ivoire": "Ivory Coast",
    "Côte d’Ivoire": "Ivory Coast",
    "IR Iran": "Iran",
    "Türkiye": "Turkey",
}

KNOCKOUT_ROUNDS = [
    "Round of 32", "Round of 16", "Quarterfinals", "Semifinals",
    "Match for third place", "Final",
]
# How many matches each round must contain. The sum is the 32 knockout games.
ROUND_SIZES = {
    "Round of 32": 16, "Round of 16": 8, "Quarterfinals": 4,
    "Semifinals": 2, "Match for third place": 1, "Final": 1,
}

_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_TEAM = (r"\[([^\]]+)\]\(https://en\.wikipedia\.org/wiki/"
         r"[^)]*?(?:national[^)]*team|national_soccer_team)[^)]*\)")
# Wikipedia separates scores with an en dash, and goal difference with a minus sign.
# Written as escapes so the intent survives any editor that normalises dashes.
_SCORE = r"\[?(\d+)\s*[\u2013\u2212-]\s*(\d+)\]?"
_ROW = re.compile(r"\|\s*" + _TEAM + r"\s*\|\s*" + _SCORE + r"(.*?)\|\s*" + _TEAM)
_ISO_DATE = re.compile(r"\((\d{4}-\d{2}-\d{2})\)")
_LONG_DATE = re.compile(r"\|\s*([A-Z][a-z]+ \d{1,2}, \d{4})\s*\|")
_GROUP_SECTION = re.compile(r"### (Group [A-L])\n")
_STANDING_ROW = re.compile(r"\|\s*([1-4])\s*\|\s*" + _TEAM + r"[^|]*\|\s*"
                           r"(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*"
                           r"(\d+)\s*\|\s*(\d+)\s*\|")


def _canon(name):
    return NAME_MAP.get(name, name)


def strip_images(text):
    """Remove inline image markdown, which otherwise sits between a team link and its
    cell delimiter and defeats every row pattern."""
    return _IMG.sub("", text)


def parse_group_results(markdown):
    """Rows for the 72 group matches, parsed from the main article's group sections.

    Each group section holds a standings table, then date headers of the form
    `| June 11, 2026 |` followed by one row per match played that day.
    """
    text = strip_images(markdown)
    sections = _GROUP_SECTION.split(text)
    rows = []
    for i in range(1, len(sections), 2):
        group = sections[i].split()[-1]
        body = sections[i + 1].split("### ")[0]
        current_date = None
        for line in body.splitlines():
            d = _LONG_DATE.search(line)
            if d and "national" not in line:
                current_date = pd.Timestamp(d.group(1)).strftime("%Y-%m-%d")
            m = _ROW.search(line)
            if m:
                rows.append({
                    "date": current_date, "stage": group,
                    "home_team": _canon(m.group(1)), "away_team": _canon(m.group(5)),
                    "home_score": int(m.group(2)), "away_score": int(m.group(3)),
                    "decided_by": "regulation",
                })
    return rows


def parse_group_standings(markdown):
    """Published standings per group: {team: (pld, w, d, l, gf, ga)}.

    Only used to cross-check the parsed match results. If the two disagree, the parse
    is wrong somewhere and the caller should refuse to write anything.
    """
    text = strip_images(markdown)
    sections = _GROUP_SECTION.split(text)
    table = {}
    for i in range(1, len(sections), 2):
        body = sections[i + 1].split("### ")[0]
        for m in _STANDING_ROW.finditer(body):
            team = _canon(m.group(2))
            table[team] = tuple(int(m.group(g)) for g in (3, 4, 5, 6, 7, 8))
    return table


def parse_knockout_results(markdown):
    """Rows for the 32 knockout matches, one section per round.

    A trailing `(a.e.t.)` in the score cell means extra time; a `Penalties` block in
    the following few lines means it went to a shootout. The recorded score is the one
    at the end of play, so a shootout reads as a draw, which is what it was.
    """
    text = strip_images(markdown)
    heading = re.compile(r"^#{2,4} (" + "|".join(re.escape(r) for r in KNOCKOUT_ROUNDS)
                         + r")\s*$", re.M)
    marks = [(m.start(), m.group(1)) for m in heading.finditer(text)]
    rows = []
    for idx, (pos, rnd) in enumerate(marks):
        end = marks[idx + 1][0] if idx + 1 < len(marks) else len(text)
        lines = text[pos:end].splitlines()
        current_date = None
        for li, line in enumerate(lines):
            d = _ISO_DATE.search(line)
            if d:
                current_date = d.group(1)
            m = _ROW.search(line)
            if m:
                middle = m.group(4) or ""
                aet = "a.e.t." in middle
                shootout = "Penalties" in "\n".join(lines[li:li + 12])
                rows.append({
                    "date": current_date, "stage": rnd,
                    "home_team": _canon(m.group(1)), "away_team": _canon(m.group(5)),
                    "home_score": int(m.group(2)), "away_score": int(m.group(3)),
                    "decided_by": ("penalties" if (aet and shootout)
                                   else "aet" if aet else "regulation"),
                })
    return rows


def standings_from_results(group_rows):
    """Recompute {team: (pld, w, d, l, gf, ga)} from parsed group matches."""
    acc = {}
    for r in group_rows:
        for team, gf, ga in ((r["home_team"], r["home_score"], r["away_score"]),
                             (r["away_team"], r["away_score"], r["home_score"])):
            pld, w, d, l, f, a = acc.get(team, (0, 0, 0, 0, 0, 0))
            acc[team] = (pld + 1, w + (gf > ga), d + (gf == ga), l + (gf < ga),
                         f + gf, a + ga)
    return acc


def validate(group_rows, knockout_rows, published_standings, canonical_teams):
    """Return a list of problems. An empty list means the parse is trustworthy.

    Every check is a hard failure rather than a warning. A silently wrong parse is the
    exact failure this project has already been bitten by once, and the standings
    cross-check is the strongest guard available: it compares the results we parsed
    against a completely separate table on the same page.
    """
    errs = []

    if len(group_rows) != 72:
        errs.append(f"expected 72 group matches, parsed {len(group_rows)}")
    per_group = {}
    for r in group_rows:
        per_group[r["stage"]] = per_group.get(r["stage"], 0) + 1
    for g, n in sorted(per_group.items()):
        if n != 6:
            errs.append(f"group {g} has {n} matches, expected 6")

    for rnd, want in ROUND_SIZES.items():
        got = sum(1 for r in knockout_rows if r["stage"] == rnd)
        if got != want:
            errs.append(f"{rnd}: parsed {got} matches, expected {want}")

    for r in group_rows + knockout_rows:
        if not r["date"]:
            errs.append(f"missing date: {r['home_team']} vs {r['away_team']}")
        for side in ("home_team", "away_team"):
            if canonical_teams and r[side] not in canonical_teams:
                errs.append(f"team name not canonical: {r[side]!r}")

    # Cross-check: results we parsed must reproduce the published group tables.
    if published_standings:
        derived = standings_from_results(group_rows)
        for team, want in sorted(published_standings.items()):
            got = derived.get(team)
            if got is None:
                errs.append(f"{team} in standings but has no parsed matches")
            elif got != want:
                errs.append(f"{team}: results give {got}, standings say {want}")

    return errs


def to_frame(group_rows, knockout_rows, source="wikipedia"):
    """Canonical results frame, ordered by date."""
    df = pd.DataFrame(group_rows + knockout_rows)
    df["match_id"] = [make_match_id(d, h, a) for d, h, a
                      in zip(df["date"], df["home_team"], df["away_team"])]
    df["source"] = source
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)[CANON_RESULT_COLS]
