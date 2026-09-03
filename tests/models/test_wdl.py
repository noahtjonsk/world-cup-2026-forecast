# tests/models/test_wdl.py
import numpy as np
import pandas as pd
import src.models.wdl  # import-smoke test: must succeed without catboost/sklearn
from src.models.wdl import (
    feature_columns, prepare_xy, predict_proba,
    train_wdl, calibrate_isotonic, NON_FEATURE_COLS,
)
from src.schema import CANON_FEATURE_COLS


def test_feature_columns_excludes_meta_and_label():
    cols = feature_columns(CANON_FEATURE_COLS)
    for meta in NON_FEATURE_COLS:
        assert meta not in cols, f"{meta!r} should be excluded"
    assert "elo_diff" in cols


def test_prepare_xy_drops_unlabeled_and_selects_features():
    base = {c: 0.0 for c in CANON_FEATURE_COLS}
    row1 = {**base, "result": "H", "elo_diff": 120.0}
    row2 = {**base, "result": None, "elo_diff": -50.0}
    df = pd.DataFrame([row1, row2])
    X, y = prepare_xy(df, label_col="result")
    assert len(X) == 1
    assert "result" not in X.columns
    assert "match_id" not in X.columns
    assert list(y) == ["H"]


def test_predict_proba_reorders_to_hda():
    class _Stub:
        classes_ = ["A", "D", "H"]

        def predict_proba(self, X):
            return [[0.1, 0.3, 0.6]]

    result = predict_proba(_Stub(), X=None)
    np.testing.assert_allclose(result, [[0.6, 0.3, 0.1]])


def test_thin_wrappers_present():
    assert callable(train_wdl)
    assert callable(calibrate_isotonic)
    # module-level import at top of file proves catboost/sklearn are NOT needed at import time
