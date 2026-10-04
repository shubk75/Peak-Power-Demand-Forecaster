#!/usr/bin/env bash
# One-time setup for a fresh git clone: venv -> packages -> pipeline -> tests.
#
# Safe to re-run: existing artifacts are kept (fixed random seeds give
# identical metrics, so re-running the pipeline changes nothing).
set -e
cd "$(dirname "$0")/.."  # repo root, wherever the script is invoked from

if [ ! -f "data/raw/dataset.csv" ]; then
    echo "data/raw/dataset.csv not found — is this a complete clone?"
    exit 1
fi

if [ -d ".venv" ]; then
    echo "=== Virtual environment already exists, skipping creation ==="
else
    echo "=== Creating virtual environment (.venv) ==="
    python3 -m venv .venv
fi

echo "=== Installing packages from requirements.txt ==="
.venv/bin/pip install -r requirements.txt

if [ -f "db/forecaster.db" ] && [ -f "models/random_forest.joblib" ]; then
    echo "=== Generated artifacts already present, skipping pipeline ==="
else
    echo "=== Running the pipeline (clean -> features -> SQLite -> train -> report) ==="
    .venv/bin/python scripts/run_pipeline.py
fi

echo "=== Running tests ==="
.venv/bin/python -m unittest discover -s tests

echo ""
echo "Setup complete. Start the dashboard with:  scripts/start.sh"
