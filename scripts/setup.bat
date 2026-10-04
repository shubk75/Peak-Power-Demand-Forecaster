@echo off
rem One-time setup for a fresh git clone: venv -> packages -> pipeline -> tests.
rem Safe to re-run: existing artifacts are kept, fixed random seeds give identical metrics.
cd /d "%~dp0.."

if not exist "data\raw\dataset.csv" (
    echo data\raw\dataset.csv not found - is this a complete clone?
    exit /b 1
)

if exist ".venv\Scripts\python.exe" (
    echo Virtual environment already exists, skipping creation.
) else (
    echo Creating virtual environment .venv ...
    python -m venv .venv
)

echo Installing packages from requirements.txt ...
.venv\Scripts\pip install -r requirements.txt

if exist "db\forecaster.db" (
    if exist "models\random_forest.joblib" (
        echo Generated artifacts already present, skipping pipeline.
        goto tests
    )
)

echo Running the pipeline - clean, features, SQLite, train, report ...
.venv\Scripts\python scripts\run_pipeline.py

:tests
echo Running tests ...
.venv\Scripts\python -m unittest discover -s tests

echo.
echo Setup complete. Start the dashboard with:  scripts\start.bat
