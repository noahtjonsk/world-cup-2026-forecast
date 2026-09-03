# tests/evaluation/test_walkforward.py
import pandas as pd
from src.evaluation.walkforward import time_splits

def test_time_splits_expanding_and_chronological():
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-01-01", "2026-02-01", "2026-03-01", "2026-04-01",
         "2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01"])})
    splits = time_splits(df, n_splits=3)                 # 4 folds of 2 -> 3 splits
    assert len(splits) == 3
    assert list(splits[0][0]) == [0, 1] and list(splits[0][1]) == [2, 3]   # train fold0, test fold1
    assert list(splits[1][0]) == [0, 1, 2, 3] and list(splits[1][1]) == [4, 5]
    assert list(splits[2][0]) == [0, 1, 2, 3, 4, 5] and list(splits[2][1]) == [6, 7]

def test_time_splits_sorts_unordered_input_no_leakage():
    df = pd.DataFrame({"date": pd.to_datetime(
        ["2026-08-01", "2026-01-01", "2026-04-01", "2026-02-01"])})        # shuffled
    (train, test), = time_splits(df, n_splits=1)                            # 2 folds of 2
    # chronological order is index [1, 3, 2, 0]; fold0=[1,3], fold1=[2,0]
    assert list(train) == [1, 3] and list(test) == [2, 0]
    assert df["date"].loc[train].max() < df["date"].loc[test].min()         # strict time order
