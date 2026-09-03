import pandas as pd
from pathlib import Path
from src.schema import CANON_MATCH_COLS
from src.ingest.results import load_results
from src.utils.io import write_parquet

def run_ingest(params, results_path=None, out_dir="data/processed"):
    """Run enabled ingestion sources, concat to canonical matches, write parquet."""
    frames = []
    if params.get("sources", {}).get("results") and results_path is not None:
        frames.append(load_results(results_path))
    # StatsBomb is ingested separately by scripts/ingest/ingest_statsbomb.py, which
    # needs network access and a competition/season list.
    matches = (pd.concat(frames, ignore_index=True)
               if frames else pd.DataFrame(columns=CANON_MATCH_COLS))
    out = Path(out_dir) / "matches.parquet"
    return write_parquet(matches, out)

def persist_tables(tables, out_dir="data/processed"):
    """Write a dict of {table_name: DataFrame} to {out_dir}/{name}.parquet."""
    from pathlib import Path
    paths = {}
    for name, df in tables.items():
        paths[name] = write_parquet(df, Path(out_dir) / f"{name}.parquet")
    return paths
