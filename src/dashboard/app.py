"""Streamlit dashboard for the Peak Power Demand Forecaster.

Read-only: loads metrics, predictions, and trained model artifacts that the
pipeline already wrote to db/forecaster.db and models/. It never retrains
models and never writes to the database.

Run from the repo root:  streamlit run src/dashboard/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))  # allow `from src...` imports when run via streamlit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data.db import DB_PATH, connect, read_features, read_metrics, read_predictions
from src.features.engineer import (
    FEATURE_COLUMNS,
    REFERENCE_SERIES,
    TEMPERATURE_COLUMNS,
    add_consecutive_hot_days,
    compute_hot_threshold,
)
from src.models.registry import load_model
from src.models.train import NAIVE_BASELINE, SPLIT_DATE


@st.cache_resource
def load_data():
    """Load everything the dashboard needs (once per process)."""
    conn = connect(DB_PATH)
    metrics = read_metrics(conn)
    predictions = read_predictions(conn)
    features = read_features(conn)
    conn.close()
    return metrics, predictions, features


def plot_actual_vs_predicted(model_pred, naive_pred, model_name):
    """Actual vs predicted demand for the selected model, test period shaded."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=model_pred.index, y=model_pred["actual_peak_mw"],
        name="Actual", line=dict(color="black", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=model_pred.index, y=model_pred["predicted_peak_mw"],
        name=f"Predicted ({model_name})", line=dict(color="crimson", width=1.5),
    ))
    fig.add_trace(go.Scatter(
        x=naive_pred.index, y=naive_pred["predicted_peak_mw"],
        name="Naive baseline", line=dict(color="gray", width=1, dash="dash"),
    ))
    fig.add_vrect(
        x0=pd.Timestamp(SPLIT_DATE), x1=model_pred.index.max(),
        fillcolor="rgba(0,0,0,0.06)", line_width=0, annotation_text="test period",
        annotation_position="top left",
    )
    fig.update_layout(
        height=420, yaxis_title="Peak demand (MW)",
        margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.08),
    )
    return fig


def plot_explainer(fitted, feature_values, model_name):
    """Feature importances (trees) or standardized coefficients (linear models)."""
    estimator = fitted.named_steps["model"] if hasattr(fitted, "named_steps") else fitted
    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        title = f"Top feature importances — {model_name}"
    else:
        values = estimator.coef_
        title = f"Standardized coefficients (top 10 by |value|) — {model_name}"
    top = sorted(zip(FEATURE_COLUMNS, values), key=lambda p: -abs(p[1]))[:10]
    labels = [p[0] for p in top][::-1]
    vals = [p[1] for p in top][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, vals, color="steelblue")
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    return fig


def plot_knn_neighbors(fitted, base_row, features):
    """The 5 historical days most similar to the chosen day (KNN explainability)."""
    scaler = fitted.named_steps["scaler"]
    knn = fitted.named_steps["model"]
    x = scaler.transform(base_row[FEATURE_COLUMNS])
    _, idx = knn.kneighbors(x, n_neighbors=5)
    train_dates = features.index[features.index < pd.Timestamp(SPLIT_DATE)]
    lines = []
    for i in idx[0]:
        date = train_dates[i]
        lines.append(f"- **{date.date()}**: {int(features['peak_mw'].loc[date]):,} MW")
    return lines


def main():
    st.set_page_config(page_title="Peak Power Demand Forecaster", layout="wide")
    metrics, predictions, features = load_data()

    if metrics.empty or not len(features):
        st.error("No trained models found. Run the pipeline first: `python run_pipeline.py`")
        st.stop()

    qualifying = metrics[metrics["qualifies_for_dashboard"] == 1]
    model_names = qualifying["model_name"].tolist()
    if not model_names:
        st.error("No model cleared the test R² > 0.5 bar. Re-run the pipeline.")
        st.stop()

    st.title("Peak Power Demand Forecaster")
    st.caption(
        "Day-ahead forecast of India's national peak electricity demand (MW), with a "
        "focus on heatwave-driven demand spikes. Built from national-grid demand + "
        "7-city temperature data."
    )

    # --- Sidebar: model selection + what-if controls -------------------------
    with st.sidebar:
        st.header("Model")
        model_name = st.selectbox(
            "Qualifying models (test R² > 0.5)", model_names,
            help="Ranked by test R². Models below 0.5 (and SVR / Decision Tree) are "
                 "excluded; the naive baseline is always shown as the benchmark.",
        )
        row = qualifying[qualifying["model_name"] == model_name].iloc[0]

        st.header("What-if heatwave")
        dates = list(features.index)
        chosen_date = st.selectbox(
            "Day", dates, index=len(dates) - 1, format_func=lambda d: str(d.date()),
        )
        delta = st.slider(
            "Temperature increase (°C)", min_value=0.0, max_value=5.0,
            step=0.5, value=1.0,
        )

    # --- Metrics panel: selected model vs naive baseline ---------------------
    naive = metrics[metrics["model_name"] == NAIVE_BASELINE].iloc[0]
    st.subheader(f"Model performance — {model_name}")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "MAE (MW)", f"{row['mae']:,.0f}",
        delta=f"{row['mae'] - naive['mae']:+,.0f} vs naive", delta_color="inverse",
    )
    col2.metric(
        "RMSE (MW)", f"{row['rmse']:,.0f}",
        delta=f"{row['rmse'] - naive['rmse']:+,.0f} vs naive", delta_color="inverse",
    )
    col3.metric(
        "Test R²", f"{row['test_r2']:.4f}",
        delta=f"{row['test_r2'] - naive['test_r2']:+.4f} vs naive",
    )
    st.caption(
        f"Metrics are computed on the held-out test period ({SPLIT_DATE} → "
        f"{features.index.max().date()}). The naive baseline predicts tomorrow's peak = "
        "today's peak; lower MAE/RMSE than the baseline is better."
    )

    # --- Actual vs predicted chart -------------------------------------------
    st.subheader("Actual vs predicted peak demand")
    model_pred = predictions[predictions["model_name"] == model_name]
    naive_pred = predictions[predictions["model_name"] == NAIVE_BASELINE]
    st.plotly_chart(plot_actual_vs_predicted(model_pred, naive_pred, model_name),
                    width="stretch")

    # --- What-if heatwave ------------------------------------------------------
    st.subheader("What-if heatwave")
    st.write(
        f"Adds **+{delta:.1f} °C** to every temperature input on "
        f"**{chosen_date.date()}** and recomputes the predicted peak demand with "
        f"**{model_name}** using the previous day's actual demand."
    )
    threshold = compute_hot_threshold(features)
    base_row = features.loc[[chosen_date]]
    bumped_frame = features.copy()
    bumped_frame.loc[chosen_date, TEMPERATURE_COLUMNS] = (
        bumped_frame.loc[chosen_date, TEMPERATURE_COLUMNS] + delta
    )
    bumped = add_consecutive_hot_days(bumped_frame, threshold)
    fitted = load_model(model_name)
    base_pred = float(fitted.predict(base_row[FEATURE_COLUMNS])[0])
    bumped_pred = float(fitted.predict(bumped.loc[[chosen_date]][FEATURE_COLUMNS])[0])
    hot_before = int(base_row["consecutive_hot_days"].iloc[0])
    hot_after = int(bumped.loc[chosen_date, "consecutive_hot_days"])

    w1, w2, w3 = st.columns(3)
    w1.metric("Predicted peak (as recorded)", f"{base_pred:,.0f} MW")
    w2.metric(
        f"Predicted peak (+{delta:.1f} °C)", f"{bumped_pred:,.0f} MW",
        delta=f"{bumped_pred - base_pred:+,.0f} MW",
    )
    w3.metric("Consecutive hot days", hot_after, delta=hot_after - hot_before)
    st.caption(
        f"A 'hot day' is a day with national aggregate max temperature ({REFERENCE_SERIES}) "
        f"above {threshold:.1f} °C — the 90th percentile of the dataset. The hot-day streak "
        "for the chosen day is recomputed after the temperature bump."
    )

    # --- Explainability --------------------------------------------------------
    st.subheader("Why the model predicts what it does")
    if model_name == "KNN":
        st.write(
            "KNN averages the 5 most similar historical days (all features, "
            "standardized). The most similar days to "
            f"{chosen_date.date()} are:"
        )
        for line in plot_knn_neighbors(fitted, base_row, features):
            st.markdown(line)
    else:
        st.pyplot(plot_explainer(fitted, base_row[FEATURE_COLUMNS], model_name))

    st.caption(
        f"Data: {len(features):,} engineered daily rows "
        f"({features.index.min().date()} → {features.index.max().date()}). "
        "This dashboard only reads from db/forecaster.db and the saved model artifacts."
    )


if __name__ == "__main__":
    main()
