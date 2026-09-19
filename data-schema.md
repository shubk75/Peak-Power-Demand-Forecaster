# Data Schema — Peak Power Demand Forecaster

## Source file
`dataset.csv` — 1,037 rows, 25 columns, daily records from **2021-08-10** to **2024-06-21**.
Demand figures are **national grid-level** (POSOCO/Grid-India via Energy Map India), not
state/city-specific. Temperature figures cover 7 Tier-1 Indian cities (Open City Portal).

## Raw columns
| Column | Type | Description |
|---|---|---|
| `date` | Date | Calendar date, YYYY-MM-DD |
| `peak_mw` | Integer | National peak power demand (MW) — **primary prediction target** |
| `shortage_mw` | Integer | Peak shortage load during demand period (MW) |
| `energy_met_mu` | Integer | Total energy delivered that day (Mega Units) |
| `energy_shortage_mu` | Float | Total energy shortage that day (MU) |
| `Delhi_TMax` / `Delhi_TMin` | Float | Delhi daily max/min temp (°C) |
| `Bengaluru_TMax` / `Bengaluru_TMin` | Float | Bengaluru daily max/min temp (°C) |
| `Chennai_TMax` / `Chennai_TMin` | Float | Chennai daily max/min temp (°C) |
| `Hyderabad_TMax` / `Hyderabad_TMin` | Float | Hyderabad daily max/min temp (°C) |
| `Kolkata_TMax` / `Kolkata_TMin` | Float | Kolkata daily max/min temp (°C) |
| `Mumbai_TMax` / `Mumbai_TMin` | Float | Mumbai daily max/min temp (°C) |
| `Pune_TMax` / `Pune_TMin` | Float | Pune daily max/min temp (°C) |
| `Natl_TMax` / `Natl_TMin` | Float | National aggregate max/min temp (°C) — reference series for heat-exposure features |
| `DayOfYear` | Integer | 1–366 |
| `Month` | Integer | 1–12 |
| `DOW` | Integer | Day of week |
| `IsWeekend` | Integer | 0/1 flag |

## Required engineered features (build these — not present in raw file)
| Feature | Definition |
|---|---|
| `peak_mw_lag1` | `peak_mw` from the previous day (`t-1`). First row will have a null — drop or backfill per `clean.py` policy, document the choice. |
| `consecutive_hot_days` | Running count of consecutive days where `Natl_TMax` (or the chosen reference series — pick one and document it) exceeds a heat threshold. Suggested starting threshold: the 90th percentile of `Natl_TMax` in the dataset, or a fixed value (e.g., 38°C) — pick one, document the choice and threshold value in code comments and the evaluation report. Resets to 0 on any day below threshold. |
| `energy_shortage_ratio` (optional but recommended) | `energy_shortage_mu / energy_met_mu` — captures grid stress independent of absolute scale. |
| Calendar features | `DayOfYear`, `Month`, `DOW`, `IsWeekend` are already present — use as-is, no need to re-derive. |

## Target variable
`peak_mw` for the **next day** (i.e., the model predicts `peak_mw(t+1)` using features known as
of day `t`, OR equivalently the table is shifted so `peak_mw(t)` is predicted from
`peak_mw_lag1` = `peak_mw(t-1)` and same-day `t` weather/calendar features — be consistent and
document which framing is used, since both are valid "day-ahead" formulations).

## Data quality checks to run before feature engineering
- No duplicate dates.
- Dates are contiguous (flag and document any gaps — do not silently interpolate demand data).
- Temperature values fall within plausible bounds for each city (e.g., 0–50°C).
- `peak_mw`, `energy_met_mu` are strictly positive.
- `shortage_mw` and `energy_shortage_mu` are ≥ 0.

## Reference comparison results (for validating the new pipeline's outputs are in the right ballpark — not a hard requirement to match exactly, since feature engineering here differs from the prior comparison)
| Model | Test R² | CV R² |
|---|---|---|
| Random Forest | 0.9131 | 0.9056 |
| Gradient Boosting | 0.8967 | 0.9006 |
| KNN | 0.8608 | 0.8674 |
| Ridge | 0.6127 | 0.6394 |
| Lasso | 0.6118 | 0.6365 |
| OLS | 0.6117 | 0.6364 |
| Elastic Net | 0.6011 | 0.6294 |
| SVR (RBF) | 0.0315 | 0.0174 |

Note: the prior comparison did not include `peak_mw_lag1` or `consecutive_hot_days` as features
(see `context.md`). Adding them is expected to change these numbers — likely improving the
linear models in particular, since lag-1 demand is typically a strong linear predictor of
next-day demand. Do not treat the table above as a target to hit exactly; treat it as a sanity
baseline.
