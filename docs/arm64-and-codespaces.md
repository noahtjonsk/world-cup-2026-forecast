# Running the socceraction tier on x64

Development happens on Windows ARM64, which cannot install a few x64-only football
libraries. Everything on the critical path works around this, but one optional module
does not, and this note records the workaround.

## What is affected

`src/deep/` wraps [socceraction](https://github.com/ML-KULeuven/socceraction) for VAEP
and xT player valuation. socceraction pins `numpy<2.0` and `pandas<2.0`, and neither has
ARM64 wheels, so the module cannot run on this machine. It is imported lazily, so
importing anything else in the project still works and the test suite still passes.

Nothing else needs it. The event metrics that actually feed the model come from
`scripts/ingest/compute_event_metrics.py`, which reads StatsBomb events through
`statsbombpy` alone and runs fine on ARM64.

`soccerdata` is blocked for a different reason, a TLS library with no ARM64 build. It is
only used by the thin wrappers in `src/ingest/elo.py` and `src/ingest/player_stats.py`,
which is why it sits in the optional `ingest` extra rather than the core dependencies.

## The workaround

GitHub Codespaces gives you a free x64 Linux VM. In its terminal:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv

python3 -m venv venv
source venv/bin/activate

# socceraction needs these exact versions
pip install numpy==1.26.4 pandas==1.5.3 pandera==0.13.4 catboost statsbombpy
pip install socceraction kloppy

python -c "import socceraction.spadl, socceraction.vaep, socceraction.xthreat; print('ok')"
```

`src/deep/pipeline.py` and `src/deep/values.py` then run there. They produce per-player
per-90 VAEP and xT metrics and per-team action-type shares, in the same schema as the
tables in `data/processed/`.

## Getting results back

Zip the parquet files and download them from the Codespaces file explorer:

```bash
zip -r deep_tier.zip data/processed/player_values.parquet data/processed/team_style.parquet
```

Drop them into `data/processed/` locally and the rest of the pipeline picks them up
through `src/utils/io.read_parquet`. No code changes needed on either side.
