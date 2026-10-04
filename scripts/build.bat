@echo off
rem Rebuild all generated artifacts (db\, models\, data\processed\, and the
rem evaluation report) from the current data\raw\dataset.csv - use this after
rem changing the dataset or the pipeline code so the dashboard reflects it.
rem Requires the one-time setup (scripts\setup.bat) to have been run.
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found at .venv\.
    echo Run the one-time setup first:  scripts\setup.bat
    exit /b 1
)

if not exist "data\raw\dataset.csv" (
    echo data\raw\dataset.csv not found - is this a complete clone?
    exit /b 1
)

echo Rebuilding artifacts from data\raw\dataset.csv ...
.venv\Scripts\python scripts\run_pipeline.py

echo Running tests ...
.venv\Scripts\python -m unittest discover -s tests

echo.
echo Rebuild complete.
