"""Build the Model page's artifacts: walk-forward metrics, calibration plot, importances.

The persisted matchup_features table holds the future 2026 fixtures, which have no
results and so cannot support a walk-forward. This builds a labeled feature table
over recent finished matches instead, then hands it to the report runner.

Lineups are left empty in that build, so the squad features come through as
missing. That keeps it fast and is worth stating plainly: the published metrics
measure the Elo, form, style and context signal, not the squad signal.

    python scripts/pipeline/build_model_artifacts.py"""
import pandas as pd

from src.simulation.format import load_tournament_format
from src.features.snapshot import build_feature_table
from src.report.artifacts import build_report_artifacts
from src.report.model_view import metrics_summary


def main(since="2023-01-01", n_splits=4):
    m = pd.read_parquet("data/processed/matches.parquet").dropna(
        subset=["home_score", "away_score"]).copy()
    m["date"] = pd.to_datetime(m["date"])
    m = m[m["date"] >= pd.Timestamp(since)].sort_values("date").reset_index(drop=True)
    rt = pd.read_parquet("data/processed/team_ratings.parquet")
    ps = pd.read_parquet("data/processed/player_stats.parquet")
    ts = pd.read_parquet("data/processed/team_style.parquet")
    hosts = load_tournament_format("2026").get("hosts", ())

    feats = build_feature_table(list(m["match_id"]), m, rt, ps, ts, pd.DataFrame(),
                                months=24, host_teams=hosts)
    print(f"built historical feature table: {len(feats)} labeled rows since {since}")

    paths = build_report_artifacts(feats, n_splits=n_splits)

    # honesty note: this build has no historical lineups, so XI/bench/role-coverage
    # features are all-NaN, append the caveat to the regenerated methodology.
    from pathlib import Path
    note = ("\n\n> Data note: the walk-forward metrics and feature importances above were "
            "computed on a lineup-free historical feature table (XI-quality, bench-dropoff "
            "and role-coverage columns all-NaN). They measure the Elo/form/style/context "
            "signal; near-zero importances for the squad features are an artifact of the "
            "build, not evidence about the squad signal.\n")
    p = Path(paths["methodology"])
    p.write_text(p.read_text(encoding="utf-8") + note, encoding="utf-8")
    print("artifacts:", paths)
    wf = pd.read_csv(paths["walkforward"])
    print("\nwalk-forward (mean over splits), CatBoost vs Elo baseline:")
    print(metrics_summary(wf).to_string(index=False))


if __name__ == "__main__":
    main()
