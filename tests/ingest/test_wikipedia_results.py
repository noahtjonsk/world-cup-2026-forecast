from src.ingest.wikipedia_results import (
    NAME_MAP, parse_group_results, parse_group_standings, parse_knockout_results,
    standings_from_results, strip_images, to_frame, validate,
)

# Two group sections in the shape the scrape actually produces: flag images sitting
# between the team link and the cell delimiter, and scores rendered as links.
GROUPS_MD = """### Group A

| Pos | Team | Pld | W | D | L | GF | GA | GD | Pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Mexico](https://en.wikipedia.org/wiki/Mexico_national_football_team "Mexico")(H) | 1 | 1 | 0 | 0 | 2 | 0 | +2 | 3 |
| 2 | [South Africa](https://en.wikipedia.org/wiki/South_Africa_national_soccer_team "SA") | 1 | 0 | 0 | 1 | 0 | 2 | -2 | 0 |

|  |  |  |  |
| --- | --- | --- | --- |
| June 11, 2026 |
| [Mexico](https://en.wikipedia.org/wiki/Mexico_national_football_team "Mexico")![](https://upload.wikimedia.org/flag.png?utm_source=x) | [2-0](https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_Group_A#a "g") | ![](https://upload.wikimedia.org/f2.png)[South Africa](https://en.wikipedia.org/wiki/South_Africa_national_soccer_team "SA") | [Estadio Azteca](https://en.wikipedia.org/wiki/Estadio_Azteca "x") |

### Group B

| Pos | Team | Pld | W | D | L | GF | GA | GD | Pts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | [Türkiye](https://en.wikipedia.org/wiki/Turkey_national_football_team "T") | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 |
| 2 | [Iran](https://en.wikipedia.org/wiki/Iran_national_football_team "I") | 1 | 0 | 1 | 0 | 1 | 1 | 0 | 1 |

|  |  |  |  |
| --- | --- | --- | --- |
| June 12, 2026 |
| [Türkiye](https://en.wikipedia.org/wiki/Turkey_national_football_team "T") | [1-1](https://en.wikipedia.org/wiki/x "g") | [Iran](https://en.wikipedia.org/wiki/Iran_national_football_team "I") | [Venue](https://en.wikipedia.org/wiki/V "v") |
"""

# Knockout rounds render the score as plain text, not a link, and mark extra time.
KNOCKOUT_MD = """## Round of 16

June29,2026(2026-06-29)

| [Canada](https://en.wikipedia.org/wiki/Canada_men's_national_soccer_team "C") | 0-3 | [Morocco](https://en.wikipedia.org/wiki/Morocco_national_football_team "M") |
| --- | --- | --- |

## Final

July19,2026(2026-07-19)

| [Spain](https://en.wikipedia.org/wiki/Spain_national_football_team "S") | 1-0 ( [a.e.t.](https://en.wikipedia.org/wiki/Overtime_(sports) "Overtime")) | [Argentina](https://en.wikipedia.org/wiki/Argentina_national_football_team "A") |
| --- | --- | --- |
"""

SHOOTOUT_MD = """## Round of 32

June29,2026(2026-06-29)

| [Germany](https://en.wikipedia.org/wiki/Germany_national_football_team "G") | 1-1 ( [a.e.t.](https://en.wikipedia.org/wiki/Overtime_(sports) "Overtime")) | [Paraguay](https://en.wikipedia.org/wiki/Paraguay_national_football_team "P") |
| --- | --- | --- |
| [Penalties](https://en.wikipedia.org/wiki/Penalty_shoot-out "pens") |
"""


def test_strip_images_removes_inline_flags():
    assert "upload.wikimedia" not in strip_images(GROUPS_MD)
    assert "Mexico" in strip_images(GROUPS_MD)


def test_parses_every_group_match_with_its_date():
    rows = parse_group_results(GROUPS_MD)
    assert len(rows) == 2
    a = rows[0]
    assert (a["home_team"], a["home_score"], a["away_score"], a["away_team"]) == \
           ("Mexico", 2, 0, "South Africa")
    assert a["date"] == "2026-06-11" and a["stage"] == "A"
    assert rows[1]["date"] == "2026-06-12" and rows[1]["stage"] == "B"


def test_wiki_spellings_map_to_canonical_names():
    # Türkiye must not survive into the table; every join downstream uses "Turkey".
    rows = parse_group_results(GROUPS_MD)
    assert rows[1]["home_team"] == "Turkey"
    assert NAME_MAP["Türkiye"] == "Turkey"


def test_score_parses_whether_or_not_it_is_a_link():
    # Group and R32 render the score as a link; R16 onward render it as plain text.
    ko = parse_knockout_results(KNOCKOUT_MD)
    assert (ko[0]["home_score"], ko[0]["away_score"]) == (0, 3)
    assert parse_group_results(GROUPS_MD)[0]["home_score"] == 2


def test_extra_time_and_shootouts_are_distinguished():
    ko = parse_knockout_results(KNOCKOUT_MD)
    final = [r for r in ko if r["stage"] == "Final"][0]
    assert final["decided_by"] == "aet"
    assert (final["home_team"], final["away_team"]) == ("Spain", "Argentina")

    pens = parse_knockout_results(SHOOTOUT_MD)[0]
    assert pens["decided_by"] == "penalties"
    # The recorded score is the one at the end of play, so a shootout reads as a draw.
    assert pens["home_score"] == pens["away_score"] == 1


def test_standings_derived_from_results_match_the_published_table():
    rows = parse_group_results(GROUPS_MD)
    published = parse_group_standings(GROUPS_MD)
    assert published["Mexico"] == (1, 1, 0, 0, 2, 0)
    assert standings_from_results(rows)["Mexico"] == published["Mexico"]
    assert standings_from_results(rows)["Turkey"] == published["Turkey"]


def test_validate_flags_a_wrong_parse_rather_than_passing_it_through():
    rows = parse_group_results(GROUPS_MD)
    published = parse_group_standings(GROUPS_MD)

    # Counts are wrong for a real tournament, so validate must complain.
    errs = validate(rows, [], published, canonical_teams=None)
    assert any("72 group matches" in e for e in errs)

    # A corrupted scoreline must be caught by the standings cross-check.
    bad = [dict(r) for r in rows]
    bad[0]["home_score"] = 5
    errs = validate(bad, [], published, canonical_teams=None)
    assert any("standings say" in e for e in errs)

    # A non-canonical team name must be caught.
    errs = validate(rows, [], {}, canonical_teams={"Mexico"})
    assert any("not canonical" in e for e in errs)


def test_frame_has_canonical_columns_and_stable_ids():
    from src.schema import CANON_RESULT_COLS
    df = to_frame(parse_group_results(GROUPS_MD), parse_knockout_results(KNOCKOUT_MD))
    assert list(df.columns) == CANON_RESULT_COLS
    assert len(df) == 4
    assert df["match_id"].nunique() == 4          # ids are unique per fixture
    assert df["date"].is_monotonic_increasing     # sorted by date
