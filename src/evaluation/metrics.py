# src/evaluation/metrics.py
import numpy as np
import pandas as pd

CLASSES = ("H", "D", "A")   # fixed order: home win / draw / away win

def _one_hot(y_true, classes):
    idx = {c: i for i, c in enumerate(classes)}
    y = np.asarray([idx[str(v)] for v in y_true])
    oh = np.zeros((len(y), len(classes)))
    oh[np.arange(len(y)), y] = 1.0
    return oh, y

def log_loss(y_true, probs, classes=CLASSES, eps=1e-15):
    """Mean multiclass cross-entropy. `probs` rows align to `classes` order."""
    probs = np.clip(np.asarray(probs, dtype=float), eps, 1 - eps)
    _, y = _one_hot(y_true, classes)
    return float(-np.mean(np.log(probs[np.arange(len(y)), y])))

def brier_score(y_true, probs, classes=CLASSES):
    """Mean multiclass Brier score: mean over rows of sum_k (p_k - o_k)^2."""
    probs = np.asarray(probs, dtype=float)
    oh, _ = _one_hot(y_true, classes)
    return float(np.mean(np.sum((probs - oh) ** 2, axis=1)))

def rps(y_true, probs, classes=CLASSES):
    """Ranked Probability Score over ORDERED classes (H<D<A). Lower is better."""
    probs = np.asarray(probs, dtype=float)
    oh, _ = _one_hot(y_true, classes)
    cum_p = np.cumsum(probs, axis=1)
    cum_o = np.cumsum(oh, axis=1)
    r = probs.shape[1]
    return float(np.mean(np.sum((cum_p[:, :-1] - cum_o[:, :-1]) ** 2, axis=1) / (r - 1)))

def calibration_table(y_true, probs, classes=CLASSES, positive_class="H", n_bins=10):
    """Reliability table for one class: bin predicted prob of `positive_class`,
    return per-bin count, mean predicted prob, and observed positive fraction.
    Empty bins are dropped. Columns: bin_lower, bin_upper, n, mean_pred, frac_pos."""
    probs = np.asarray(probs, dtype=float)
    ci = list(classes).index(positive_class)
    p = probs[:, ci]
    obs = np.asarray([1.0 if str(v) == positive_class else 0.0 for v in y_true])
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    b = np.clip(np.digitize(p, edges, right=False) - 1, 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        rows.append({"bin_lower": float(edges[k]), "bin_upper": float(edges[k + 1]),
                     "n": int(m.sum()), "mean_pred": float(p[m].mean()),
                     "frac_pos": float(obs[m].mean())})
    return pd.DataFrame(rows, columns=["bin_lower", "bin_upper", "n", "mean_pred", "frac_pos"])
