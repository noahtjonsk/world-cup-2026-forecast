import pandas as pd
from src.utils.io import write_parquet, read_parquet

def test_parquet_roundtrip_creates_dirs(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    out = tmp_path / "nested" / "df.parquet"
    write_parquet(df, out)
    assert out.exists()
    back = read_parquet(out)
    pd.testing.assert_frame_equal(df, back)
