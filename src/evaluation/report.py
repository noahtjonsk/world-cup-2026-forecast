# src/evaluation/report.py
import numpy as np
import pandas as pd
from src.evaluation.metrics import log_loss, rps, brier_score, calibration_table
from src.evaluation.walkforward import time_splits
from src.models.elo_baseline import predict_proba as elo_predict, fit_draw_base
from src.models.wdl import prepare_xy, train_wdl, predict_proba as cat_predict


def walkforward_compare(features, n_splits=4):
    feats = features.dropna(subset=["result"]).reset_index(drop=True)
    rows = []
    for i, (tr, te) in enumerate(time_splits(feats, n_splits=n_splits)):
        train, test = feats.loc[tr], feats.loc[te]
        y_test = test["result"].to_numpy()
        db = fit_draw_base(train["elo_diff"], train["neutral"], train["result"])
        p_elo = elo_predict(test["elo_diff"], test["neutral"], draw_base=db)
        rows.append({"model": "elo", "split": i, "log_loss": log_loss(y_test, p_elo),
                     "rps": rps(y_test, p_elo), "brier": brier_score(y_test, p_elo)})
        Xtr, ytr = prepare_xy(train)
        Xte, _ = prepare_xy(test)
        model = train_wdl(Xtr, ytr)
        p_cat = cat_predict(model, Xte)
        rows.append({"model": "catboost", "split": i, "log_loss": log_loss(y_test, p_cat),
                     "rps": rps(y_test, p_cat), "brier": brier_score(y_test, p_cat)})
    return pd.DataFrame(rows)


def plot_calibration(y_true, probs, out_path="reports/calibration.png", positive_class="H"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path
    tbl = calibration_table(y_true, probs, positive_class=positive_class)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfect")
    ax.plot(tbl["mean_pred"], tbl["frac_pos"], "o-", label=f"P({positive_class})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title(f"Calibration: {positive_class}")
    ax.legend()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path
