# src/models/wdl.py
import numpy as np

NON_FEATURE_COLS = ("match_id", "date", "home_team", "away_team", "snapshot_date", "result")


def feature_columns(columns):
    return [c for c in columns if c not in NON_FEATURE_COLS]


def prepare_xy(features, label_col="result"):
    df = features.dropna(subset=[label_col]).copy()
    return df[feature_columns(df.columns)], df[label_col]


def predict_proba(model, X, classes=("H", "D", "A")):
    proba = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = [str(c) for c in model.classes_]
    order = [model_classes.index(c) for c in classes]
    return proba[:, order]


def train_wdl(X, y, cat_features=None, **kwargs):
    from catboost import CatBoostClassifier
    params = dict(loss_function="MultiClass", iterations=500, depth=6,
                  learning_rate=0.05, random_seed=42, verbose=False)
    params.update(kwargs)
    model = CatBoostClassifier(**params)
    model.fit(X, y, cat_features=cat_features or [])
    return model


def calibrate_isotonic(probs, y_true, classes=("H", "D", "A")):
    from sklearn.isotonic import IsotonicRegression
    probs = np.asarray(probs, dtype=float)
    out = np.zeros_like(probs)
    for k, c in enumerate(classes):
        target = np.asarray([1.0 if str(v) == c else 0.0 for v in y_true])
        out[:, k] = IsotonicRegression(out_of_bounds="clip").fit_transform(probs[:, k], target)
    row = out.sum(axis=1, keepdims=True)
    return np.divide(out, row, out=np.full_like(out, 1.0 / len(classes)), where=row > 0)
