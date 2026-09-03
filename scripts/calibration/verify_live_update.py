"""End-to-end check of the after-match update loop, writing to a temporary directory.

Verifies three things: a full cycle produces per-fixture predictions whose W/D/L
sums to one with plausible expected goals; re-running leaves the ratings file
byte-identical, since the Elo replay is deterministic; and the rerun simulation
produces title odds. Nothing under data/processed is touched.
Records the outcome to reports/live_update_check.md."""
import hashlib
import time
from pathlib import Path


from src.utils.io import read_parquet
from src.config import load_params
from src.simulation.format import load_tournament_format
from src.simulation.montecarlo import champion_odds
from src.states.elo_update import seed_from_ratings
from src.states.runner import run_live_update, COARSE_UPDATE_NOTE


def _md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def main():
    out = Path("data/tmp_smoke")
    out.mkdir(parents=True, exist_ok=True)

    rt = read_parquet("data/processed/team_ratings.parquet")
    m = read_parquet("data/processed/matches.parquet")
    fx = read_parquet("data/processed/fixtures.parquet")
    ps = read_parquet("data/processed/player_stats.parquet")
    ts = read_parquet("data/processed/team_style.parquet")

    cut = load_params()["states"]["live_cutoff_date"]
    seed = seed_from_ratings(rt, cut)
    fmt = load_tournament_format("2026")

    t0 = time.time()
    preds = run_live_update(m, fx, seed, fmt, ps, ts, base_ratings=rt, out_dir=str(out))
    h1 = _md5(out / "ratings.parquet")
    t1 = time.time()

    # idempotency: re-run, ratings must be byte-identical
    run_live_update(m, fx, seed, fmt, ps, ts, base_ratings=rt, out_dir=str(out))
    h2 = _md5(out / "ratings.parquet")

    cols = ["home_team", "away_team", "p_home", "p_draw", "p_away",
            "exp_goals_home", "exp_goals_away"]
    psum = (preds[["p_home", "p_draw", "p_away"]].sum(axis=1))
    odds = champion_odds(read_parquet(str(out / "simulation_results.parquet")), top=8)

    lines = []
    lines.append("# Live-update engine: end-to-end check")
    lines.append("")
    lines.append(f"- COARSE note carried: {COARSE_UPDATE_NOTE!r}")
    lines.append(f"- upcoming fixtures forecast: {len(preds)}")
    lines.append(f"- W/D/L row sums in [0.999, 1.001]: {bool(((psum - 1).abs() < 1e-3).all())}")
    lines.append(f"- expected goals range: home [{preds['exp_goals_home'].min():.2f}, "
                 f"{preds['exp_goals_home'].max():.2f}], away "
                 f"[{preds['exp_goals_away'].min():.2f}, {preds['exp_goals_away'].max():.2f}]")
    lines.append(f"- IDEMPOTENT (ratings.parquet md5 identical on re-run): {h1 == h2}  "
                 f"({h1} == {h2})")
    lines.append(f"- first cycle wall time: {t1 - t0:.1f}s")
    lines.append("")
    lines.append("## match_predictions (head 10)")
    lines.append("")
    lines.append("```")
    lines.append(preds[cols].head(10).to_string(index=False))
    lines.append("```")
    lines.append("")
    lines.append("## title odds (top 8)")
    lines.append("")
    lines.append("```")
    lines.append(odds.to_string(index=False))
    lines.append("```")
    report = "\n".join(lines)

    Path("reports").mkdir(exist_ok=True)
    Path("reports/live_update_check.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
