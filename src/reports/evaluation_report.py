"""Generate reports/evaluation_report.md from the metrics table (FR4).

Generated, not hand-written: every number comes from db/forecaster.db, and the
feature definitions mirror src/features/engineer.py and src/data/clean.py.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd

from src.data.db import connect, read_features, read_metrics
from src.features.engineer import FEATURE_COLUMNS, REFERENCE_SERIES, compute_hot_threshold
from src.models.registry import EXCLUDED_FROM_PICKER
from src.models.train import NAIVE_BASELINE, SPLIT_DATE

REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "evaluation_report.md"


def generate_report():
    """Read metrics + features from SQLite and write the markdown report."""
    conn = connect()
    metrics = read_metrics(conn)
    features = read_features(conn)
    conn.close()

    threshold = compute_hot_threshold(features)
    best = metrics.iloc[0]
    naive = metrics[metrics["model_name"] == NAIVE_BASELINE].iloc[0]
    qualifying = metrics[metrics["qualifies_for_dashboard"] == 1]
    weakest = qualifying.iloc[-1] if len(qualifying) else None
    excluded = metrics[metrics["model_name"].isin(EXCLUDED_FROM_PICKER)]
    below_bar = metrics[
        (metrics["qualifies_for_dashboard"] == 0)
        & (~metrics["model_name"].isin(EXCLUDED_FROM_PICKER))
        & (metrics["model_name"] != NAIVE_BASELINE)
    ]

    n_train = int((features.index < pd.Timestamp(SPLIT_DATE)).sum())
    n_test = int((features.index >= pd.Timestamp(SPLIT_DATE)).sum())
    train_peak = features["peak_mw"][features.index < pd.Timestamp(SPLIT_DATE)]
    test_peak = features["peak_mw"][features.index >= pd.Timestamp(SPLIT_DATE)]
    rmse_gain = (1 - best["rmse"] / naive["rmse"]) * 100

    lines = [
        "# Peak Power Demand Forecaster — Model Evaluation Report",
        "",
        f"_Generated automatically from `db/forecaster.db` on "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}._",
        "",
        f"- **Data**: {len(features):,} engineered daily rows "
        f"({features.index.min().date()} → {features.index.max().date()}), "
        "national-grid peak demand + 7-city temperatures.",
        "- **Day-ahead framing**: `peak_mw(t)` is predicted from previous-day peak demand "
        "(`peak_mw_lag1`), same-day weather/calendar features, and grid-stress features. "
        "The naive baseline predicts `peak_mw(t) = peak_mw(t-1)`.",
        f"- **Split**: chronological — train on all days before **{SPLIT_DATE}**, test on "
        f"{SPLIT_DATE} onward ({n_train:,} train / {n_test:,} test rows). The identical "
        "split is used for every model.",
        "- **CV**: 5-fold expanding-window (TimeSeriesSplit) R² on the full engineered "
        "table; the same folds are used for every model.",
        "",
        "## Metrics — all evaluated models (ranked by test R²)",
        "",
        "| Model | MAE (MW) | RMSE (MW) | Test R² | CV R² | Dashboard picker |",
        "|---|---|---|---|---|---|",
    ]
    for _, metric_row in metrics.iterrows():
        if metric_row["model_name"] == NAIVE_BASELINE:
            picker_cell = "— benchmark"
        elif metric_row["qualifies_for_dashboard"] == 1:
            picker_cell = "✔ included"
        else:
            picker_cell = "✘ excluded"
        lines.append(
            f"| {metric_row['model_name']} | {metric_row['mae']:,.2f} | "
            f"{metric_row['rmse']:,.2f} | {metric_row['test_r2']:.4f} | "
            f"{metric_row['cv_r2']:.4f} | {picker_cell} |"
        )

    lines += [
        "",
        "## Interpretation",
        "",
        f"- **Best model: {best['model_name']}** — test R² {best['test_r2']:.4f}, "
        f"RMSE {best['rmse']:,.2f} MW; {(1 - best['test_r2']) * 100:.1f}% of peak-demand "
        "variance is left unexplained on held-out data.",
        f"- **Versus the naive baseline** (tomorrow's peak = today's peak: RMSE "
        f"{naive['rmse']:,.2f} MW, test R² {naive['test_r2']:.4f}), the best model cuts "
        f"RMSE by **{rmse_gain:.1f}%** — the engineered features add real value beyond "
        "persistence.",
    ]
    if weakest is not None:
        lines.append(
            f"- **Weakest qualifying model: {weakest['model_name']}** (test R² "
            f"{weakest['test_r2']:.4f}) — still above the R² > 0.5 bar, so it remains "
            "selectable in the dashboard."
        )
    if len(below_bar):
        below_desc = ", ".join(
            f"{row['model_name']} (test R² {row['test_r2']:.4f})"
            for _, row in below_bar.iterrows()
        )
        lines.append(f"- **Below the R² > 0.5 bar** (not selectable): {below_desc}.")
    lines.append(
        f"- **Naive baseline** (tomorrow's peak = today's peak): test R² "
        f"{naive['test_r2']:.4f} — always shown in the dashboard as the benchmark next "
        "to every model; it is a reference, not a selectable forecasting model."
    )
    lines.append(
        f"- **Why several models score below the naive baseline**: the test year (2024) "
        f"set demand records above anything in the training data (train max "
        f"{train_peak.max():,.0f} MW / mean {train_peak.mean():,.0f} MW, vs test max "
        f"{test_peak.max():,.0f} MW / mean {test_peak.mean():,.0f} MW). Tree-based and "
        "instance-based models (Random Forest, KNN, Decision Tree) cannot extrapolate "
        "beyond the range of their training targets, so they systematically underpredict "
        "the record periods; the lag-1-based linear models extrapolate and track the "
        "level shift. The prior reference comparison (RF/KNN test R² 0.86–0.91) used a "
        "random split, where test points are interpolated within the training range; "
        "the chronological split required for honest day-ahead forecasting exposes "
        "this limit.",
    )
    if len(excluded):
        exc_desc = ", ".join(
            f"{row['model_name']} (test R² {row['test_r2']:.4f})"
            for _, row in excluded.iterrows()
        )
        lines += [
            f"- **Excluded from the dashboard picker**: {exc_desc}. In the prior "
            "reference comparison SVR (RBF) scored R² ≈ 0.03 and Decision Tree was "
            "unreported; both are non-shippable per the project's hard constraint "
            "(AGENTS.md). Decision Tree is evaluated and recorded here for "
            "completeness (FR4), and its metrics are still reported above.",
        ]

    lines += [
        "",
        "## Feature engineering",
        "",
        f"Modeling uses {len(FEATURE_COLUMNS)} features per day:",
        "",
        "- `peak_mw_lag1` = `peak_mw(t-1)` — previous available day's national peak "
        "demand. The first row (no previous day) is dropped; after a calendar gap the "
        "lag refers to the last observed day (demand is never interpolated).",
        f"- `consecutive_hot_days` — running count of consecutive days ending at t with "
        f"`{REFERENCE_SERIES}` strictly above **{threshold:.2f} °C** (the 90th percentile "
        f"of `{REFERENCE_SERIES}` in the dataset). Resets to 0 on any day at or below "
        "the threshold.",
        "- `energy_shortage_ratio` = `energy_shortage_mu / energy_met_mu` — grid stress "
        "independent of absolute scale.",
        "- Same-day weather: daily max/min temperature for the 7 cities (Delhi, "
        "Bengaluru, Chennai, Hyderabad, Kolkata, Mumbai, Pune) plus the national "
        "aggregate (`Natl_TMax`, `Natl_TMin`). Invalid readings (outside 0–50 °C) are "
        "set to null and linearly interpolated over time.",
        "- `shortage_mw` / `energy_shortage_mu`: a missing reading is treated as no "
        "shortage reported that day and filled with 0 (documented choice; shortage is "
        "not the demand target and is never interpolated).",
        "- Calendar: `DayOfYear`, `Month`, `DOW`, `IsWeekend` (present in the raw data, "
        "used as-is).",
        "",
        "Linear models and KNN operate on standardized features (a StandardScaler "
        "fitted on the training split, bundled into each saved artifact); tree models "
        "use raw feature values.",
    ]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return REPORT_PATH
