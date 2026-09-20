#!/usr/bin/env bash
set -e
.venv/bin/python run_pipeline.py
.venv/bin/streamlit run src/dashboard/app.py
