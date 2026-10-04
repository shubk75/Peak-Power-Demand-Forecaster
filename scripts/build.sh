#!/usr/bin/env bash
# Rebuild all generated artifacts (db/, models/, data/processed/, and the
# evaluation report) from the current data/raw/dataset.csv — use this after
# changing the dataset or the pipeline code so the dashboard reflects it.
#
# Requires the one-time setup (scripts/setup.sh) to have been run.
set -e
cd "$(dirname "$0")/.."  # repo root, wherever the script is invoked from

if [ ! -x ".venv/bin/python" ]; then
    echo "No virtual environment found at .venv/."
    echo "Run the one-time setup first:  scripts/setup.sh"
    exit 1
fi

if [ ! -f "data/raw/dataset.csv" ]; then
    echo "data/raw/dataset.csv not found — is this a complete clone?"
    exit 1
fi

echo "=== Rebuilding artifacts from data/raw/dataset.csv ==="
.venv/bin/python scripts/run_pipeline.py

echo "=== Running tests ==="
.venv/bin/python -m unittest discover -s tests

echo ""
echo "Rebuild complete."
