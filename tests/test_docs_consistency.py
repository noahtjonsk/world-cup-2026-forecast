"""Guards against documentation drifting away from the code it describes.

Every failure this file has caught so far was the same shape: a count or a list that
was correct when written and quietly stopped being true. The README has claimed the
wrong number of tests, described five dashboard pages when there were six, and shipped
a Home page that did not link to one of them.

Checks that need the generated parquet tables skip when those are absent, since
data/processed is gitignored and a fresh clone will not have it.
"""
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
DESIGN = (ROOT / "docs/design.md").read_text(encoding="utf-8")
PROCESSED = ROOT / "data/processed"

PAGE_WORDS = {4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight",
              9: "nine", 10: "ten"}


def _pages():
    return sorted(p.name for p in (ROOT / "app/pages").glob("*.py"))


def test_readme_test_count_matches_the_suite(request):
    """The README quotes a test total in two places. Both must be true.

    Uses pytest's own collection count rather than counting `def test_` lines, because
    parametrized tests expand into several and a naive count undershoots. Skips on a
    filtered run, where the collected total is a subset and proves nothing.
    """
    collected = request.session.testscollected
    if collected < 100:
        pytest.skip("partial run; the count only means something for the whole suite")
    quoted = {int(n) for n in re.findall(r"(\d+)[- ]tests?\b", README)}
    assert quoted, "README no longer quotes a test count; drop this test or restore it"
    assert quoted == {collected}, (
        f"README quotes {sorted(quoted)} tests, the suite collects {collected}"
    )


def test_home_links_every_dashboard_page():
    home = (ROOT / "app/Home.py").read_text(encoding="utf-8")
    linked = set(re.findall(r'page_link\("pages/([^"]+)"', home))
    missing = set(_pages()) - linked
    assert not missing, f"Home.py does not link: {sorted(missing)}"


def _page_names():
    """Human names of the dashboard pages, from `1_Match.py` to `Match`."""
    return [p.stem.split("_", 1)[1] for p in sorted((ROOT / "app/pages").glob("*.py"))]


def test_readme_describes_every_page_by_name():
    """Matches page names rather than counting, so reflowing a paragraph in the
    hard-wrapped README cannot fail this for the wrong reason."""
    flat = " ".join(README.split())
    missing = [n for n in _page_names() if f"The {n} page " not in flat]
    assert not missing, f"README does not describe: {missing}"


def test_every_page_has_a_screenshot_named_after_it():
    """Keyed on page name, so adding a diagram or any other image to docs/img does
    not fail this and send the reader looking in the wrong place."""
    shots = {p.stem.lower() for p in (ROOT / "docs/img").glob("*.png")}
    missing = [n for n in _page_names() if n.lower() not in shots]
    assert not missing, f"no screenshot in docs/img for: {missing}"


def test_readme_states_the_right_number_of_pages():
    n = len(_pages())
    word = PAGE_WORDS.get(n)
    assert word is not None, (
        f"{n} dashboard pages exist but PAGE_WORDS in this test file only covers "
        f"{sorted(PAGE_WORDS)}; extend the mapping here, not the README"
    )
    assert f"{word} pages" in README, f"README should say '{word} pages'"


def test_app_page_count_assertion_tracks_the_pages():
    """tests/app/test_app_pages.py hardcodes a file count; keep it honest."""
    src = (ROOT / "tests/app/test_app_pages.py").read_text(encoding="utf-8")
    asserted = int(re.search(r"len\(files\) == (\d+)", src).group(1))
    assert asserted == len(_pages()) + 1, "count should be the pages plus Home.py"


def test_design_doc_parameters_match_the_config():
    cfg = yaml.safe_load((ROOT / "config/params.yaml").read_text(encoding="utf-8"))
    dc = cfg["models"]["dixon_coles"]
    expected = {
        "prior_strength": dc["prior_strength"], "prior_scale": dc["prior_scale"],
        "elo_anchor_weight": dc["elo_anchor_weight"], "squad_coef": dc["squad_coef"],
        "half_life_days": dc["half_life_days"],
        "sim_runs": cfg["sim_runs"], "sim_jitter": cfg["sim_jitter"],
    }
    for key, value in expected.items():
        row = re.search(r"\| `" + re.escape(key) + r"`[^|]*\| ([^|]+) \|", DESIGN)
        assert row, f"docs/design.md has no table row for {key}"
        stated = row.group(1).strip().replace(",", "")
        assert stated == str(value), f"{key}: design.md says {stated}, config says {value}"


@pytest.mark.parametrize("doc", ["README.md", "docs/design.md",
                                 "docs/arm64-and-codespaces.md"])
def test_documented_script_paths_exist(doc):
    text = (ROOT / doc).read_text(encoding="utf-8")
    for script in sorted(set(re.findall(r"(scripts/[\w/]+\.py)", text))):
        assert (ROOT / script).exists(), f"{doc} references missing {script}"


@pytest.mark.parametrize("doc", ["README.md", "docs/design.md",
                                 "docs/arm64-and-codespaces.md"])
def test_relative_links_resolve(doc):
    base = (ROOT / doc).parent
    text = (ROOT / doc).read_text(encoding="utf-8")
    for target in re.findall(r"\]\((?!https?:)([^)#]+)\)", text):
        assert (base / target).exists() or (ROOT / target).exists(), \
            f"{doc} links to missing {target}"


def test_no_em_or_en_dashes_in_prose():
    """House style, and a reliable tell when text is pasted in from elsewhere."""
    for doc in ("README.md", "docs/design.md", "docs/arm64-and-codespaces.md"):
        text = (ROOT / doc).read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text, f"{doc} contains a dash"


@pytest.mark.skipif(not (PROCESSED / "matches.parquet").exists(),
                    reason="generated tables absent; they are gitignored")
def test_readme_data_table_row_counts_are_current():
    """The README documents the snapshot the published forecast was built on.

    Tables the project pins itself must match exactly. Tables fed by a live upstream
    dataset are checked as a lower bound instead: martj42's results file and the FBref
    scrapes both grow over time, so a rebuild today legitimately yields more rows than
    the June 2026 ingest did. The error worth catching is the README claiming more data
    than exists, not the upstream having moved on.
    """
    from src.utils.io import read_parquet
    pinned = {"2026 fixtures": "fixtures", "2026 groups": "groups",
              "Predicted lineups": "lineups"}
    upstream = {"Historical matches": "matches", "Team Elo ratings": "team_ratings",
                "Player stats": "player_stats", "Player event values": "player_values",
                "Team style profiles": "team_style"}

    for label, table in {**pinned, **upstream}.items():
        row = re.search(r"\| " + re.escape(label) + r" \| ([\d,]+) \|", README)
        assert row, f"README data table has no row for {label}"
        stated = int(row.group(1).replace(",", ""))
        actual = len(read_parquet(PROCESSED / f"{table}.parquet"))
        if label in pinned:
            assert stated == actual, f"{label}: README says {stated:,}, table has {actual:,}"
        else:
            assert actual >= stated, (
                f"{label}: README claims {stated:,} rows but the table has only "
                f"{actual:,}; the data has shrunk, which usually means a partial ingest "
                f"overwrote it"
            )


@pytest.mark.skipif(not (PROCESSED / "results_2026.parquet").exists(),
                    reason="results absent; run scripts/ingest/fetch_2026_results.py")
def test_readme_scorecard_figures_match_recomputation():
    from src.evaluation import scorecard as sc
    from src.utils.io import read_parquet
    joined = sc.join_predictions(read_parquet(PROCESSED / "match_predictions.parquet"),
                                 read_parquet(PROCESSED / "results_2026.parquet"))
    metrics = sc.wdl_metrics(joined)
    published = metrics[metrics["model"].str.startswith("published")].iloc[0]
    skill = sc.skill_interval(joined)

    for pattern, actual in [
        (r"\| The published forecast \| (\d+\.\d+)", published["log_loss"]),
        (r"by (\d+\.\d+) in log-loss per match", skill["mean_advantage"]),
        (r"interval of (\d+\.\d+) to", skill["ci_low"]),
        (r"interval of \d+\.\d+ to (\d+\.\d+)", skill["ci_high"]),
    ]:
        m = re.search(pattern, README)
        assert m, f"README no longer states the figure matching {pattern!r}"
        assert abs(float(m.group(1)) - actual) < 0.0006, \
            f"README says {m.group(1)}, recomputation gives {actual:.4f}"
