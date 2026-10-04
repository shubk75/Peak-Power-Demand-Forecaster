@echo off
rem Start the Peak Power Demand Forecaster dashboard server.
rem Does not retrain anything - run scripts\setup.bat first if the venv or the
rem generated artifacts (db\, models\) are missing.
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
    echo No virtual environment found at .venv\.
    echo Run the one-time setup first:  scripts\setup.bat
    exit /b 1
)

if not exist "db\forecaster.db" (
    echo Generated artifacts missing - run the one-time setup first:  scripts\setup.bat
    exit /b 1
)

if not exist "models\random_forest.joblib" (
    echo Generated artifacts missing - run the one-time setup first:  scripts\setup.bat
    exit /b 1
)

echo Starting dashboard at http://localhost:8501  (Ctrl+C to stop)
.venv\Scripts\streamlit run src\dashboard\app.py
