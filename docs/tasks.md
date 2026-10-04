# Build Plan — Peak Power Demand Forecaster

Work through phases in order. Each phase has a Definition of Done (DoD) — do not start the next
phase until the current one's DoD is met. Check items off as you complete them.

## Phase 0 — Project scaffold
- [x] Create the folder layout exactly as in `architecture.md`.
- [x] Add `requirements.txt` (pandas, numpy, scikit-learn, streamlit, matplotlib, plotly, joblib).
- [x] Copy `dataset.csv` into `data/raw/`.
- [x] Initialize git, first commit.

**DoD:** `pip install -r requirements.txt` succeeds; folder structure matches `architecture.md`.

## Phase 1 — Data ingestion & cleaning
- [x] `src/data/load.py`: read `data/raw/dataset.csv`, validate columns/types against
  `data-schema.md`.
- [x] `src/data/clean.py`: run all data-quality checks listed in `data-schema.md`; fix or
  clearly document any issues found (no silent fixes).
- [x] Write a small script/notebook cell that prints row count, date range, and a null-count
  summary, and confirm it matches `data-schema.md` (1,037 rows, 2021-08-10 to 2024-06-21).

**DoD:** cleaned DataFrame loads with no unexplained nulls; date range and row count confirmed.
**Done (2026-09-19):** 1,037 rows confirmed; issues found & documented — 2 invalid temperature
readings (> 50 °C, interpolated), 2 missing shortage readings (filled 0), 10 calendar gaps
(reported, not filled), 4 suspected copy-paste demand rows (flagged, kept).

## Phase 2 — Feature engineering
- [x] `src/features/engineer.py`: implement `peak_mw_lag1`, `consecutive_hot_days` (document
  chosen threshold and reference series), `energy_shortage_ratio`, and pass through calendar
  columns.
- [x] Decide and document the day-ahead framing (see `data-schema.md` — target variable section).
- [x] Write `data/processed/features.csv`.
- [x] `tests/test_features.py`: unit tests for `consecutive_hot_days` (e.g., a hand-built
  mini-series with known runs of hot days) and for `peak_mw_lag1` alignment.

**DoD:** `features.csv` exists, tests pass, feature definitions match `data-schema.md`.
**Done (2026-09-19):** framing = shift framing (`peak_mw(t)` from `peak_mw_lag1` = `peak_mw(t-1)`
+ same-day weather/calendar); reference series = `Natl_TMax`, threshold = 90th percentile
(≈ 36.59 °C); 1,036 rows × 23 features.

## Phase 3 — Database layer
- [x] `src/data/db.py`: create `db/forecaster.db` with the `features`, `predictions`, `metrics`
  tables per `architecture.md`.
- [x] Load `features.csv` into the `features` table.

**DoD:** `db/forecaster.db` exists and `features` table row count matches `features.csv`.
**Done (2026-09-19):** 1,036 rows in both; INSERT names columns explicitly so column order
can never silently swap values (regression-tested in `tests/test_db.py`).

## Phase 4 — Modeling
- [x] `src/models/train.py`: implement a single chronological train/test split (document the
  split date/ratio) used identically by all 7 models: OLS, Ridge, Lasso, Elastic Net, KNN,
  Random Forest, Gradient Boosting. Fix random seeds for Random Forest and Gradient Boosting.
- [x] Also train the naive baseline (`predicted(t) = actual(t-1)`, i.e. reuse `peak_mw_lag1`
  directly as the prediction) for comparison.
- [x] `src/models/evaluate.py`: compute MAE, RMSE, test R², and 5-fold CV R² for each of the 7
  models + naive baseline.
- [x] `src/models/registry.py`: mark each model `qualifies_for_dashboard = True` if test R² > 0.5,
  else `False`. Save each trained model to `models/` via joblib.
- [x] Write predictions and metrics into the `predictions` and `metrics` tables.
- [x] `tests/test_models.py`: assert all 7 models trained without error, assert metrics table has
  exactly 8 rows (7 models + naive baseline), assert qualifying flag logic is correct given the
  computed R² values.

**DoD:** `metrics` table populated for all 8 rows; at least the models expected to clear R² > 0.5
per `data-schema.md`'s reference table (Ridge, Lasso, OLS, Elastic Net, KNN, Random Forest,
Gradient Boosting) are flagged correctly given this pipeline's actual results — if the added
lag/heat features change which models qualify, that's fine, just make sure the flag reflects the
real computed R², not the reference table.

**Done (2026-09-19):** split = train before 2024-01-01 (863 rows) / test 2024 (173 rows), fixed
seeds 42. `tests/test_models.py` asserts the 8 required rows are all present (the table holds 10
rows: + SVR and Decision Tree, evaluated & recorded per requirements.md FR4, both excluded from
the picker). Actual qualifying set with the chronological split: **OLS (0.7624), Ridge (0.7622),
Lasso (0.7630), Gradient Boosting (0.5693)** — Random Forest (0.3414), KNN (−2.5120) and
Elastic Net (0.0013) fall below the bar because the 2024 test year sets demand records above the
training range and non-extrapolating/strongly-regularized models underpredict it (see the
evaluation report). Flags reflect the real computed R².

## Phase 5 — Evaluation report
- [x] `src/reports/evaluation_report.py`: read `metrics` table, generate `docs/reports/
  evaluation_report.md` with a ranked table (by test R²) and a short written interpretation
  (best model, weakest qualifying model, why SVR/Decision Tree — if trained — are excluded,
  comparison against naive baseline).

**DoD:** `docs/reports/evaluation_report.md` is generated (not hand-written) and matches the DB
contents.

## Phase 6 — Dashboard
- [x] `src/dashboard/app.py`: Streamlit app with:
  - Model selector limited to `qualifies_for_dashboard = True` models.
  - Actual-vs-predicted line chart for the selected model over the test period.
  - Metrics panel (MAE/RMSE/R² for selected model vs. naive baseline).
  - "What-if" temperature slider (+0°C to +5°C) applied to a user-selected date's temperature
    input, recomputing the predicted `peak_mw` live using the selected model.
- [x] Confirm `streamlit run src/dashboard/app.py` launches without retraining any model (reads
  only from `db/forecaster.db` and `models/`).

**DoD:** dashboard runs locally, model picker only shows qualifying models, what-if slider
updates the prediction without a full page reload delay of more than ~1s.
**Done (2026-09-19):** verified via `streamlit.testing.v1.AppTest` — picker shows exactly the
qualifying models (Lasso, OLS, Ridge, Gradient Boosting), metrics panel compares against the
naive baseline, what-if slider recomputes live (models/DB loaded once via `st.cache_resource`).
The chart covers the full historical period with the test window shaded (requirements.md FR5).

## Phase 7 — Polish & handoff
- [x] README.md at repo root: how to set up, run the pipeline end-to-end, and launch the
  dashboard.
- [x] Confirm every file in `architecture.md`'s layout exists and is used (delete anything
  unused).
- [x] Final commit.

**DoD:** a fresh clone + `pip install -r requirements.txt` + documented commands reproduces the
DB, models, report, and dashboard from scratch.
