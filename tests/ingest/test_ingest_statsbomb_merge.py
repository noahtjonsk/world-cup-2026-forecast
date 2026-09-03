"""The merge logic that stops a StatsBomb refresh from truncating matches.parquet.

An earlier version of the script wrote only its own rows, which cut the table from
49,555 rows to 334. These pin the behaviour that prevents a repeat.
"""
import pandas as pd

from scripts.ingest.ingest_statsbomb import merge_into_existing
from src.schema import CANON_MATCH_COLS


def _row(match_id, source, home="A", away="B"):
    return {"match_id": match_id, "date": pd.Timestamp("2024-01-01"),
            "competition": "X", "season": "2024", "home_team": home, "away_team": away,
            "home_score": 1, "away_score": 0, "stage": None, "neutral": False,
            "source": source}


def test_merge_keeps_rows_from_other_sources():
    existing = pd.DataFrame([_row("a", "results"), _row("b", "results"),
                             _row("c", "statsbomb")])[CANON_MATCH_COLS]
    fetched = pd.DataFrame([_row("c", "statsbomb"), _row("d", "statsbomb")])[CANON_MATCH_COLS]
    merged = merge_into_existing(fetched, existing)
    assert len(merged) == 4                                  # 2 results + 2 statsbomb
    assert set(merged[merged["source"] == "results"]["match_id"]) == {"a", "b"}


def test_merge_replaces_only_the_statsbomb_rows():
    existing = pd.DataFrame([_row("a", "results"), _row("old", "statsbomb")])[CANON_MATCH_COLS]
    fetched = pd.DataFrame([_row("new", "statsbomb")])[CANON_MATCH_COLS]
    merged = merge_into_existing(fetched, existing)
    ids = set(merged["match_id"])
    assert "old" not in ids and "new" in ids and "a" in ids


def test_merge_on_an_empty_table_just_uses_the_fetch():
    fetched = pd.DataFrame([_row("a", "statsbomb")])[CANON_MATCH_COLS]
    assert len(merge_into_existing(fetched, None)) == 1
    assert len(merge_into_existing(fetched, pd.DataFrame())) == 1


def test_merge_preserves_canonical_columns():
    existing = pd.DataFrame([_row("a", "results")])[CANON_MATCH_COLS]
    fetched = pd.DataFrame([_row("b", "statsbomb")])[CANON_MATCH_COLS]
    assert list(merge_into_existing(fetched, existing).columns) == CANON_MATCH_COLS


def test_importing_the_script_does_no_work():
    """The original had no entry-point guard, so importing it rewrote the table."""
    import inspect

    import scripts.ingest.ingest_statsbomb as mod
    src = inspect.getsource(mod)
    assert '__name__ == "__main__"' in src, "the script must guard its entry point"
    body = src.split('if __name__ == "__main__"')[0]
    assert "persist_tables(" not in body.replace("    persist_tables(", ""), \
        "no write may happen at module level"
