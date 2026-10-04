"""One-command pipeline: raw CSV -> clean -> features -> SQLite -> models -> report.

Run from the repo root:  python scripts/run_pipeline.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `from src...` imports when run via scripts/

from src.data.clean import clean
from src.data.db import connect, write_features, write_metrics, write_predictions
from src.data.load import load_raw
from src.features.engineer import FEATURE_COLUMNS, build_features, compute_hot_threshold
from src.models.evaluate import evaluate_all, predictions_frame
from src.models.registry import flag_metrics_rows, save_model
from src.models.train import train_all
from src.reports.evaluation_report import generate_report

PROCESSED_CSV = Path(__file__).resolve().parents[1] / "data" / "processed" / "features.csv"


def main():
    """Run every phase and print a QA summary as it goes."""
    print("=== Peak Power Demand Forecaster — pipeline ===")

    # Phase 1 — load + validate, and print the data summary.
    raw = load_raw()
    nulls = raw.isna().sum()
    nulls = nulls[nulls > 0]
    print(f"raw rows: {len(raw)}")
    print(f"date range: {raw.index.min().date()} -> {raw.index.max().date()}")
    print(f"nulls (non-zero columns): {nulls.to_dict() or 'none'}")

    data, report = clean(raw)
    print(
        f"clean rows: {report.n_rows_clean} "
        f"({report.n_duplicate_dates} duplicate dates dropped, "
        f"{len(report.dropped_demand_dates)} invalid demand rows dropped)"
    )
    if report.invalid_temp_dates:
        print(f"invalid temperatures repaired by interpolation: {report.invalid_temp_dates}")
    if report.shortage_filled_dates:
        print(
            "missing shortage readings treated as no shortage (filled 0): "
            f"{report.shortage_filled_dates}"
        )
    if report.suspected_duplicate_demand_dates:
        print(
            "suspected copy-paste demand rows (flagged, kept): "
            f"{report.suspected_duplicate_demand_dates}"
        )
    if report.date_gaps:
        print(f"calendar gaps (reported, not filled): {report.n_date_gaps} missing dates")
    if report.n_negative_shortage:
        print(f"WARNING: {report.n_negative_shortage} negative shortage values found")

    # Phase 2 — feature engineering.
    threshold = compute_hot_threshold(data)
    features = build_features(data, hot_threshold=threshold)
    PROCESSED_CSV.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_CSV)
    print(
        f"features: {len(features)} rows x {len(FEATURE_COLUMNS)} modeling features "
        f"-> {PROCESSED_CSV}"
    )
    print(f"hot-day threshold: {threshold:.2f} degC (90th percentile of Natl_TMax)")

    # Phase 3 — database.
    conn = connect()
    write_features(conn, features)
    print(f"db: features table written ({len(features)} rows)")

    # Phase 4 — train + evaluate all models, save artifacts, persist predictions.
    fitted = train_all(features)
    for name, model in fitted.items():
        save_model(model, name)
    metrics = flag_metrics_rows(evaluate_all(features, fitted))
    write_predictions(conn, predictions_frame(fitted, features))
    write_metrics(conn, metrics.set_index("model_name"))
    conn.close()
    print("db: predictions + metrics tables written; model artifacts saved")
    print(metrics.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # Phase 5 — evaluation report.
    path = generate_report()
    print(f"report: {path}")
    print("=== done ===")


if __name__ == "__main__":
    main()
