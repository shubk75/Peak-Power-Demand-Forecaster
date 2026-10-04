"""Evaluation: MAE/RMSE/R2 on the test set plus 5-fold CV R2, and the naive baseline.

CV policy: TimeSeriesSplit(n_splits=5) — expanding-window CV, the appropriate
configuration for chronological data. The same folds are used for every model
and the naive baseline. RMSE is computed as sqrt(MSE) so results are identical
across sklearn versions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

from src.features.engineer import FEATURE_COLUMNS
from src.models.train import NAIVE_BASELINE, chronological_split, make_model

N_CV_SPLITS = 5


def regression_metrics(y_true, y_pred):
    """MAE, RMSE and R2 for one set of predictions."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "test_r2": float(r2_score(y_true, y_pred)),
    }


def cross_val_r2(model, X, y):
    """Mean R2 across the shared 5-fold time-series CV for an unfitted model."""
    scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(N_CV_SPLITS), scoring="r2")
    return float(scores.mean())


def naive_cv_r2(X, y):
    """Mean CV R2 for the naive baseline (prediction = lag-1 demand)."""
    scores = []
    for _, test_idx in TimeSeriesSplit(N_CV_SPLITS).split(X):
        naive_pred = X["peak_mw_lag1"].iloc[test_idx]
        scores.append(r2_score(y.iloc[test_idx], naive_pred))
    return float(np.mean(scores))


def evaluate_all(features, fitted_models):
    """Metrics for every fitted model plus the naive baseline, as a DataFrame."""
    _, X_test, _, y_test = chronological_split(features)
    X_full = features[FEATURE_COLUMNS]
    y_full = features["peak_mw"]
    rows = []
    for name, model in fitted_models.items():
        rows.append({
            "model_name": name,
            **regression_metrics(y_test, model.predict(X_test)),
            "cv_r2": cross_val_r2(make_model(name), X_full, y_full),
        })
    rows.append({
        "model_name": NAIVE_BASELINE,
        **regression_metrics(y_test, X_test["peak_mw_lag1"]),
        "cv_r2": naive_cv_r2(X_full, y_full),
    })
    return pd.DataFrame(rows)


def predictions_frame(fitted_models, features):
    """Per-model predictions for every date in the engineered table.

    Predictions are stored for the full period (train + test) so the dashboard
    can chart actual-vs-predicted over the whole historical record; reported
    metrics use only the held-out test portion.
    """
    X = features[FEATURE_COLUMNS]
    frames = [
        pd.DataFrame({
            "date": features.index,
            "model_name": name,
            "actual_peak_mw": features["peak_mw"].to_numpy(),
            "predicted_peak_mw": model.predict(X),
        })
        for name, model in fitted_models.items()
    ]
    return pd.concat(frames, ignore_index=True).set_index("date")
