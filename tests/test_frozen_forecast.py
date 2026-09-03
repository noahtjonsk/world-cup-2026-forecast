"""Makes the forecast freeze real rather than conventional.

The published prediction is only worth anything because it predates the tournament.
Several scripts write to these same paths: `build_dashboard_data.py` persists
match_predictions, and `run_live_cycle.py` and `run_simulation.py` rewrite the
simulation outputs. One accidental run replaces an out-of-sample forecast with a
hindsight one, and every number on the Scorecard page, in
reports/forecast_scorecard.md and in the README silently becomes false together.

Nothing detected that before this test existed. The risk is not hypothetical: during
the review a script with no entry-point guard truncated matches.parquet from 49,555
rows to 334 simply by being imported.

If this fails, do not update the checksums to make it pass. Work out which run
overwrote the file and restore it. The checksums live in frozen_forecast.sha256.json
and should change only when the forecast is deliberately reissued, which for a
published pre-tournament prediction means never.
"""
import hashlib
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).parent / "frozen_forecast.sha256.json"
EXPECTED = json.loads(MANIFEST.read_text(encoding="utf-8"))


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("relative_path", sorted(EXPECTED))
def test_published_forecast_artifact_is_unchanged(relative_path):
    path = ROOT / relative_path
    if not path.exists():
        # The parquet tables are gitignored, so a fresh clone legitimately lacks them.
        # reports/simulation.md is committed and must always be there.
        if relative_path.startswith("data/"):
            pytest.skip(f"{relative_path} absent; generated tables are not in the repo")
        pytest.fail(f"{relative_path} is committed and must exist")

    actual = _digest(path)
    assert actual == EXPECTED[relative_path], (
        f"{relative_path} has changed.\n"
        f"  expected {EXPECTED[relative_path]}\n"
        f"  found    {actual}\n"
        "This file is the published pre-tournament forecast. Something rewrote it, most "
        "likely run_simulation.py, run_live_cycle.py or build_dashboard_data.py. Restore "
        "it rather than updating the checksum."
    )


def test_manifest_covers_every_published_artifact():
    """A new frozen output must be added here, or it goes unguarded."""
    assert set(EXPECTED) == {
        "data/processed/match_predictions.parquet",
        "data/processed/simulation_results.parquet",
        "data/processed/bracket_results.parquet",
        "reports/simulation.md",
    }
