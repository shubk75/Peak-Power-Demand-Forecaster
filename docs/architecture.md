# Architecture — Peak Power Demand Forecaster

## Tech stack
| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.x | Single language across pipeline, models, DB, UI |
| Data handling | pandas, NumPy | CSV parsing, cleaning, feature engineering |
| Modeling | scikit-learn | OLS, Ridge, Lasso, Elastic Net, KNN, Random Forest, Gradient Boosting — one consistent API |
| Storage | SQLite (`sqlite3`, stdlib) | Zero-config, single-file, sufficient for ~1,000 rows |
| Dashboard | Streamlit + Matplotlib/Plotly | Fast interactive UI without a separate frontend |
| Persistence of models | `joblib` (or `pickle`) | Save trained models/predictions so the dashboard loads instead of retrains |

## Repository layout
```
peak-power-demand-forecaster/
├── AGENTS.md
├── README.md
├── requirements.txt
├── docs/                          # all documentation
│   ├── requirements.md
│   ├── data-schema.md
│   ├── architecture.md
│   ├── tasks.md
│   └── reports/
│       └── evaluation_report.md   # generated output, not hand-written
├── scripts/
│   ├── run_pipeline.py            # one-command pipeline entrypoint
│   ├── setup.sh / setup.bat       # one-time setup for a fresh clone
│   └── start.sh / start.bat       # start the Streamlit dashboard server
├── data/
│   ├── raw/
│   │   └── dataset.csv            # source, unmodified
│   └── processed/
│       └── features.csv           # cleaned + engineered, staging before DB load
├── db/
│   └── forecaster.db              # SQLite: raw table, features table, predictions table, metrics table
├── src/
│   ├── data/
│   │   ├── load.py                # read + validate dataset.csv
│   │   ├── clean.py               # missing values, sanity checks
│   │   └── db.py                  # SQLite read/write helpers
│   ├── features/
│   │   └── engineer.py            # lag features, consecutive-hot-days, calendar features
│   ├── models/
│   │   ├── train.py                # trains all 7 models on identical split
│   │   ├── evaluate.py             # MAE/RMSE/R², naive baseline, CV
│   │   └── registry.py             # save/load trained models + which ones qualify (R² > 0.5)
│   ├── reports/
│   │   └── evaluation_report.py   # generates the metrics table + written summary
│   └── dashboard/
│       └── app.py                  # Streamlit entrypoint
├── models/                         # saved model artifacts (joblib files)
└── tests/
    ├── test_features.py
    ├── test_models.py
    └── test_db.py
```

## Data flow
1. `data/raw/dataset.csv` → `src/data/load.py` → `src/data/clean.py` → `data/processed/features.csv`
   (via `src/features/engineer.py`).
2. `data/processed/features.csv` → `src/data/db.py` writes a `features` table into
   `db/forecaster.db`.
3. `src/models/train.py` reads from `db/forecaster.db`, trains all 7 models on one fixed
   train/test split (chronological split recommended — train on earlier dates, test on later
   dates, since this is a time-series forecasting problem), and writes:
   - trained model artifacts → `models/`
   - per-model predictions on the test set → `predictions` table in `db/forecaster.db`
   - per-model metrics (MAE, RMSE, R², CV R²) → `metrics` table in `db/forecaster.db`, including
     a `naive_baseline` row.
4. `src/reports/evaluation_report.py` reads the `metrics` table and writes
   `docs/reports/evaluation_report.md`.
5. `src/dashboard/app.py` reads `metrics`, `predictions`, and loads model artifacts from
   `models/` for the live what-if slider. It does not retrain models.

## SQLite schema (tables)
- `features` — one row per date: all columns from `data-schema.md`'s engineered feature set,
  primary key `date`.
- `predictions` — `date, model_name, actual_peak_mw, predicted_peak_mw`.
- `metrics` — `model_name, mae, rmse, test_r2, cv_r2, qualifies_for_dashboard (bool)`.

## Module boundaries
- **`src/data/`** never trains models or touches Streamlit.
- **`src/features/`** is pure functions on DataFrames — no I/O beyond what's passed in, so it's
  independently testable.
- **`src/models/`** never imports Streamlit.
- **`src/dashboard/app.py`** never trains models or writes to the DB — it only reads.

## Environment
Matches the original project's target environment: Intel Core i5 / AMD Ryzen 5 or better, 8 GB
RAM minimum, Python 3.x, no GPU required.
