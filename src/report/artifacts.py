from pathlib import Path


def build_report_artifacts(features, reports_dir="reports", n_splits=4):
    """Regenerate the Model page's artifacts from a labeled feature table.

    Runs the walk-forward comparison, plots the calibration curve, and fits CatBoost
    once more for global feature importance. Writes walkforward.csv, calibration.png,
    feature_importance.csv and methodology.md into `reports_dir` and returns their
    paths.

    Not fixture-tested, since it needs optional dependencies and writes files. Covered
    by an import smoke test and by scripts/pipeline/build_model_artifacts.py. scipy,
    scikit-learn and matplotlib are all imported lazily."""
    import pandas as pd
    from src.evaluation.report import walkforward_compare, plot_calibration
    from src.models.wdl import prepare_xy, train_wdl, feature_columns, predict_proba
    from src.report.model_view import metrics_summary, METHODOLOGY, LIMITATIONS

    out = Path(reports_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. walk-forward metrics: CatBoost vs the Elo baseline
    wf = walkforward_compare(features, n_splits=n_splits)
    wf.to_csv(out / "walkforward.csv", index=False)

    # 2. global feature importance from a single fit on everything
    X, y = prepare_xy(features)
    model = train_wdl(X, y)
    imp = pd.DataFrame({
        "feature": feature_columns(features.columns),
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    imp.to_csv(out / "feature_importance.csv", index=False)

    # 3. calibration curve for the home-win class
    proba = predict_proba(model, X)
    plot_calibration(proba, y, out_path=str(out / "calibration.png"), positive_class="H")

    # 4. methodology and limitations writeup; model_view is the single source of truth
    summary = metrics_summary(wf)
    lines = ["# World Cup 2026 forecast: methodology and limitations", "",
             METHODOLOGY, "", "## Walk-forward metrics (mean over splits)", "",
             "| model | log_loss | rps | brier |", "|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(f"| {r['model']} | {r['log_loss']:.4f} | {r['rps']:.4f} | {r['brier']:.4f} |")
    # LIMITATIONS opens with its own label line; that becomes the heading, so drop it
    # from the body rather than printing it twice.
    body = LIMITATIONS.split("\n", 1)[1]
    lines += ["", "## Limitations (read these before trusting any number)", "", body]
    (out / "methodology.md").write_text("\n".join(lines), encoding="utf-8")

    return {k: str(out / v) for k, v in {
        "walkforward": "walkforward.csv", "feature_importance": "feature_importance.csv",
        "calibration": "calibration.png", "methodology": "methodology.md"}.items()}
