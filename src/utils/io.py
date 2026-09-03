from pathlib import Path
import pandas as pd

def write_parquet(df, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p

def read_parquet(path):
    return pd.read_parquet(path)
