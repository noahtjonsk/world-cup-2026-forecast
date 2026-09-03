import pandas as pd
from src.models.goals import score_matrix


def match_card(pred, rho=0.0, max_goals=10, top_scorelines=5):
    """Render-ready dict for one match_predictions row (CANON_PREDICTION_COLS slice):
    W/D/L, expected goals, the Dixon-Coles `scoreline_grid` (pure score_matrix on the
    row's expected goals), and the `top_scorelines` [( (h, a), prob ), ...] sorted
    desc. No model recompute and no optional dep, score_matrix is pure numpy."""
    M = score_matrix(float(pred["exp_goals_home"]), float(pred["exp_goals_away"]),
                     rho=rho, max_goals=max_goals)
    cells = [((i, j), float(M[i, j])) for i in range(M.shape[0]) for j in range(M.shape[1])]
    cells.sort(key=lambda kv: kv[1], reverse=True)
    return {
        "home_team": pred["home_team"], "away_team": pred["away_team"],
        "p_home": float(pred["p_home"]), "p_draw": float(pred["p_draw"]),
        "p_away": float(pred["p_away"]),
        "exp_goals_home": float(pred["exp_goals_home"]),
        "exp_goals_away": float(pred["exp_goals_away"]),
        "scoreline_grid": M,
        "top_scorelines": cells[:top_scorelines],
    }


def top_drivers(feature_row, importances, k=5):
    """The k highest-importance features present in this match's feature row.

    `importances` has columns feature and importance. Each returned row pairs a
    feature name with its value for this match.

    These describe what the model leaned on, not what caused the result. A feature can
    rank high because it correlates with something the model cannot see."""
    rows = []
    for _, r in importances.sort_values("importance", ascending=False).iterrows():
        f = r["feature"]
        if f in feature_row:
            rows.append({"feature": f, "importance": float(r["importance"]),
                         "value": float(feature_row[f])})
        if len(rows) >= k:
            break
    return pd.DataFrame(rows, columns=["feature", "importance", "value"])
