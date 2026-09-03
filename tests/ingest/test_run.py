from src.ingest.run import run_ingest
from src.utils.io import read_parquet
from src.schema import CANON_MATCH_COLS

def test_run_ingest_writes_canonical_matches(tmp_path):
    csv = tmp_path / "results.csv"
    csv.write_text(
        "date,home_team,away_team,home_score,away_score,tournament,city,country,neutral\n"
        "2024-06-15,Germany,Scotland,5,1,UEFA Euro,Munich,Germany,False\n"
    )
    params = {"sources": {"results": True, "statsbomb": False}}
    out = run_ingest(params, results_path=csv, out_dir=tmp_path / "processed")
    df = read_parquet(out)
    assert out.exists()
    assert list(df.columns) == CANON_MATCH_COLS
    assert len(df) == 1

import pandas as pd
from src.ingest.run import persist_tables

def test_persist_tables_writes_each_named_table(tmp_path):
    tables = {
        "team_ratings": pd.DataFrame({"date": ["2026-03-01"], "team": ["Brazil"], "elo": [2100.0], "source": ["eloratings"]}),
        "injuries": pd.DataFrame({"date": ["2026-06-10"], "team": ["Spain"], "player": ["Gavi"], "reason": ["Knee"], "status": ["Missing Fixture"], "source": ["apifootball"]}),
    }
    paths = persist_tables(tables, out_dir=tmp_path / "processed")
    assert set(paths) == {"team_ratings", "injuries"}
    assert read_parquet(paths["team_ratings"]).loc[0, "team"] == "Brazil"
    assert paths["injuries"].exists()
