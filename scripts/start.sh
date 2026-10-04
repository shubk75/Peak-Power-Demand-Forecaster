#!/usr/bin/env bash
# Start the Peak Power Demand Forecaster dashboard server.
#
# Does not retrain anything — run scripts/setup.sh first if the venv or the
# generated artifacts (db/, models/) are missing.
set -e
cd "$(dirname "$0")/.."  # repo root, wherever the script is invoked from

if [ ! -x ".venv/bin/python" ]; then
    echo "No virtual environment found at .venv/."
    echo "Run the one-time setup first:  scripts/setup.sh"
    exit 1
fi

if [ ! -f "db/forecaster.db" ] || [ ! -f "models/random_forest.joblib" ]; then
    echo "Generated artifacts missing (db/forecaster.db, models/*.joblib)."
    echo "Run the one-time setup first:  scripts/setup.sh"
    exit 1
fi

echo "Starting dashboard at http://localhost:8501  (Ctrl+C to stop)"
exec .venv/bin/streamlit run src/dashboard/app.py
