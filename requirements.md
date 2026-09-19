# Requirements — Peak Power Demand Forecaster

## Objective
Given historical national-grid peak demand and 7-city temperature data, predict **next-day
national peak electricity demand (MW)**, with a focus on accuracy during heatwave-driven demand
spikes, and expose the result through an interactive dashboard usable by a non-technical grid
planner.

## Functional requirements

### FR1 — Data pipeline
- Load `dataset.csv` (1,037 rows, 10 Aug 2021–21 Jun 2024) and validate schema against
  `data-schema.md`.
- Clean: handle any missing values, confirm date continuity, confirm numeric ranges are sane
  (e.g., temperatures within plausible bounds, MW values positive).
- Persist the cleaned, feature-engineered table into a SQLite database (see `architecture.md`).

### FR2 — Feature engineering
- Derive lag features (at minimum lag-1 peak demand).
- Derive a consecutive-hot-days count (e.g., using `Natl_TMax` or a chosen reference city above
  a heat threshold).
- Retain `DayOfYear`, `Month`, `DOW`, `IsWeekend` as seasonal/calendar features.
- Document every derived feature's definition and formula in code comments/docstrings and in
  the evaluation report.

### FR3 — Modeling
- Train these 7 models on the engineered feature set, using an identical train/test split (and
  optionally 5-fold CV) for all of them: **OLS, Ridge, Lasso, Elastic Net, KNN, Random Forest,
  Gradient Boosting**.
- Compute MAE, RMSE, and R² (test-set and CV) for each model.
- Compute the same metrics for the naive baseline ("tomorrow = today's peak demand").
- Only models with **test R² > 0.5** are eligible for the dashboard's model picker.
- Save trained models (or their parameters/predictions) so the dashboard does not retrain on
  every load.

### FR4 — Evaluation report
- Produce a metrics table (all 7 models + naive baseline) ranked by test R².
- Written interpretation: best model, weakest qualifying model, and why SVR/Decision Tree are
  excluded (reference figures: SVR R² ≈ 0.03; Decision Tree not evaluated in the prior
  comparison — this project should evaluate it too and record its result, but it still doesn't
  ship in the dashboard unless it clears R² > 0.5).

### FR5 — Dashboard (Streamlit)
- Chart: actual vs. predicted peak demand over the historical period, for the selected model.
- Model selector: dropdown limited to qualifying models (R² > 0.5).
- Metrics panel: MAE/RMSE/R² for the selected model, shown alongside the naive baseline for
  comparison.
- "What-if" temperature slider: lets the user add +1°C to +5°C to a chosen day's temperature
  input and see the recomputed predicted peak demand.
- Should run locally via `streamlit run app.py` (exact path per `architecture.md`).

## Non-functional requirements
- **Reproducibility**: fixed random seeds for any stochastic model (Random Forest, Gradient
  Boosting) so re-running the pipeline gives consistent metrics.
- **Explainability**: prefer models and features whose behavior can be explained to a grid
  planner (e.g., feature importances for tree-based models, coefficients for linear models)
  over a fully opaque model.
- **Performance**: full pipeline (data load → features → train all 7 models → metrics) should
  run in well under a minute on a standard laptop (Intel i5/Ryzen 5, 8 GB RAM) given the dataset
  size (~1,000 rows).
- **Portability**: no cloud dependencies required to run; SQLite file and CSV should be
  self-contained in the repo/working directory.

## Explicit non-goals (for this build)
- No live/real-time data ingestion from POSOCO/IMD APIs — `dataset.csv` is the fixed source.
- No state- or city-level demand forecasting — the target variable is national-grid `peak_mw`.
- No deep learning models — scope is limited to the 7 regression algorithms listed above.

## Success criteria
- Pipeline runs end-to-end from raw CSV to a working dashboard without manual intervention.
- At least the models matching the reference comparison (Ridge, Lasso, OLS, Elastic Net, KNN,
  Random Forest, Gradient Boosting) are trained, evaluated, and — where R² > 0.5 — selectable in
  the dashboard.
- Dashboard correctly reflects each selected model's stored metrics and updates the "what-if"
  prediction live.
