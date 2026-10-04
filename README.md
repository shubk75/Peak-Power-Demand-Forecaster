# Peak Power Demand Forecaster

Day-ahead forecasting of **India's national peak electricity demand (MW)**, with a focus on
heatwave-driven demand spikes, served through an interactive Streamlit dashboard.

Given historical national-grid demand and 7-city temperature data, the system predicts
next-day national peak demand, compares 9 regression models (plus a naive persistence
baseline), and lets a non-technical grid planner explore "what-if" heatwave scenarios.

## Setup

One-time setup for a fresh clone (creates the venv, installs packages, runs the pipeline and
tests):

```bash
scripts/setup.sh          # Linux / macOS
scripts\setup.bat         # Windows
```

Or manually:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python scripts/run_pipeline.py
```

Python 3.x, no GPU required. Everything runs locally — no cloud dependencies.

## Run

```bash
# 1. Full pipeline: raw CSV -> clean -> features -> SQLite -> train all models -> report
.venv/bin/python scripts/run_pipeline.py

# 2. Tests
.venv/bin/python -m unittest discover -s tests

# 3. Dashboard (reads only from db/ and models/ — never retrains)
scripts/start.sh
```

Or `scripts/build.sh` (Windows: `scripts\build.bat`) — rebuilds all artifacts from the current
`data/raw/dataset.csv` (pipeline + tests) in one go; use it after changing the dataset or the
pipeline code so the dashboard reflects the changes.

`scripts/start.sh` (or `scripts\start.bat` on Windows) starts the Streamlit server — it checks
the venv and generated artifacts first and points you to `scripts/setup.sh` if anything is
missing.

A fresh clone + `scripts/setup.sh` reproduces the feature table, SQLite database, saved models,
and evaluation report from scratch.

## What the pipeline produces

| Artifact | Path | Purpose |
|---|---|---|
| Feature table | `data/processed/features.csv` | cleaned + engineered, staging before DB load |
| SQLite DB | `db/forecaster.db` | `features`, `predictions`, `metrics` tables |
| Model artifacts | `models/*.joblib` | trained models (scaler bundled for linear/KNN) |
| Evaluation report | `docs/reports/evaluation_report.md` | generated metrics table + written interpretation |

Generated artifacts are git-ignored (they are reproducible from a fresh clone via the commands
above, with fixed random seeds giving identical metrics) — run `scripts/setup.sh` (or
`python scripts/run_pipeline.py`) once after cloning to create them.

## Repository layout

```
├── AGENTS.md                     # build spec for AI coding agents
├── requirements.txt
├── docs/                         # all documentation
│   ├── requirements.md           # what must be true when this is done
│   ├── data-schema.md            # exact data + features to derive
│   ├── architecture.md           # folder layout, module boundaries, tech stack
│   ├── tasks.md                  # phased build plan
│   └── reports/
│       └── evaluation_report.md  # generated metrics table + interpretation
├── scripts/
│   ├── run_pipeline.py           # one-command pipeline entrypoint
│   ├── build.sh / build.bat      # rebuild artifacts from the current dataset
│   ├── setup.sh / setup.bat      # one-time setup for a fresh clone
│   └── start.sh / start.bat      # start the Streamlit dashboard server
├── data/
│   ├── raw/dataset.csv          # source data (1,037 days, 10 Aug 2021 – 21 Jun 2024)
│   └── processed/features.csv   # generated
├── db/forecaster.db             # generated
├── src/
│   ├── data/                    # load.py, clean.py, db.py
│   ├── features/engineer.py     # lag-1 demand, consecutive hot days, shortage ratio
│   ├── models/                  # train.py, evaluate.py, registry.py
│   ├── reports/evaluation_report.py
│   └── dashboard/app.py         # Streamlit entrypoint
├── models/                      # generated artifacts
└── tests/                       # test_features.py, test_models.py, test_db.py
```

## Modeling decisions (short version — details in the evaluation report)

- **Day-ahead framing**: `peak_mw(t)` is predicted from previous-day peak demand
  (`peak_mw_lag1`), same-day weather/calendar features, and grid-stress features. The naive
  baseline predicts `peak_mw(t) = peak_mw(t-1)`.
- **Split**: chronological — train on all days before 2024-01-01, test on 2024 onward. The
  identical split is used by every model, so comparisons stay apples-to-apples.
- **Heat threshold**: `consecutive_hot_days` counts days with `Natl_TMax` strictly above the
  90th percentile of the dataset (≈ 36.59 °C).
- **Dashboard model picker**: only models with test R² > 0.5 (SVR/Decision Tree additionally
  excluded per project constraints; the naive baseline is shown as a benchmark, not a
  selectable model).

## Data provenance

`data/raw/dataset.csv` is the fixed source: national-grid demand (POSOCO/Grid-India via
Energy Map India) and 7-city temperatures (Open City Portal). It is never re-scraped or
substituted. The `files/` directory (if present) holds reference/context documents only and
is git-ignored; the pipeline never reads from it.
