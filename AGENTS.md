# AGENTS.md — Peak Power Demand Forecaster

You are building **Peak Power Demand Forecaster**, a day-ahead Indian national-grid electricity
peak demand forecasting system that predicts heatwave-driven demand spikes, deployed as an
interactive Streamlit dashboard.

Read these files in this order before writing code, and keep them in sync as you build:
1. `requirements.md` — what must be true when this is done
2. `data-schema.md` — the exact data you have and the features you must derive
3. `architecture.md` — folder layout, module boundaries, tech stack
4. `tasks.md` — the phased build plan; work through it top to bottom, checking items off

## Non-negotiable constraints
- **Multi-model, not single-model.** Train and expose OLS, Ridge, Lasso, Elastic Net, KNN,
  Random Forest, and Gradient Boosting. Only ship/select models whose test R² exceeds 0.5 in
  the final dashboard's model picker; still report every model's metrics in the evaluation
  report even if a model is excluded from the picker. Do not hardcode Random Forest as "the"
  model — the dashboard must let the user pick among the qualifying models.
- **Benchmark against a naive baseline** ("tomorrow's peak = today's peak") in every metrics
  report and in the dashboard, not just against each other.
- **Source data is `dataset.csv`** (already provided, 1,037 rows, 10 Aug 2021–21 Jun 2024,
  national-grid-level demand + 7-city temperatures). Do not re-scrape or substitute a different
  dataset unless a task explicitly says to supplement it.
- **Engineered features are required**, not optional: lag-1 demand, consecutive-hot-days count,
  and any others listed in `data-schema.md`. A model trained only on raw same-day temperature is
  not acceptable as a final deliverable.
- **Persist to SQLite**, staging through CSV/pandas as the intermediate format, per
  `architecture.md`.
- **Dashboard must include**: actual-vs-predicted demand chart, RMSE/R² comparison table across
  all evaluated models, model-selection control, and a "what-if" temperature slider (+1°C to
  +5°C) that recomputes the predicted peak demand live.

## Conventions
- Python 3.x. pandas + NumPy for data work, scikit-learn for the 7 regression models,
  Streamlit + Matplotlib/Plotly for the UI, `sqlite3` (stdlib) for storage.
- Keep data ingestion, feature engineering, modeling, and UI in separate modules — see
  `architecture.md` for the exact layout. Do not put model training code inside the Streamlit
  file.
- Every model must be evaluated with the same train/test split (or the same 5-fold CV
  configuration) so comparisons in `regression_model_comparative_analysis` stay apples-to-apples.
- Write short docstrings, not long comments. Prefer readable, explicit code over cleverness —
  this is a student PBL project that will be read and graded, not a production SaaS.
- After each phase in `tasks.md`, run whatever tests/checks that phase specifies before moving
  to the next phase.

## Do not
- Do not invent electricity demand or temperature data beyond `dataset.csv`.
- Do not silently drop the naive-baseline comparison from the final report or dashboard.
- Do not ship SVR (RBF) or Decision Tree as selectable models — both are below or lack a
  qualifying R² in the reference comparison (SVR ≈ 0.03; Decision Tree unreported). They may
  still be trained for completeness in the evaluation script if a task asks for it, but must not
  appear in the dashboard's model picker.
- Do not skip writing predictions/metrics to SQLite — the dashboard should read from the DB, not
  recompute models on every page load.

## Reference source documents (context only, do not treat as literal file paths in this repo)
`context.md` (already produced) has the full project background, team, original proposal, and
the 9-model comparison results this project is built on. Use it for narrative/report sections;
the four docs above are the operative build spec.
