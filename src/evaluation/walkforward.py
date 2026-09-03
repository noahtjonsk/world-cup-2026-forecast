# src/evaluation/walkforward.py
import numpy as np
import pandas as pd

def time_splits(df, date_col="date", n_splits=3):
    """Expanding-window walk-forward splits, strictly chronological (no shuffle).

    Sorts rows by `date_col`, partitions them into `n_splits + 1` contiguous
    folds, and for split i yields (train = folds 0..i, test = fold i+1). Returns
    a list of (train_index, test_index) numpy arrays of df index LABELS, so
    callers use `df.loc[train]` / `df.loc[test]`."""
    order = pd.to_datetime(df[date_col]).sort_values(kind="mergesort").index.to_numpy()
    folds = np.array_split(order, n_splits + 1)
    return [(np.concatenate(folds[: i + 1]), folds[i + 1]) for i in range(n_splits)]
