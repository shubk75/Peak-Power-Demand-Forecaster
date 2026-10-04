"""Train all regression models on one identical chronological split.

Split policy (documented, used by every model and the baseline): train on all
days before 2024-01-01 (~83% of rows), test on 2024-01-01 onward (~17%).
Because this is a time-series forecasting problem the split is chronological,
never random. Random seeds are fixed for the stochastic models (Random Forest,
Gradient Boosting) so re-runs give identical metrics.

Linear models and KNN are wrapped in a StandardScaler pipeline: each saved
artifact carries its own scaler, which the dashboard's what-if slider relies
on. Tree models are scale-invariant and are saved as-is.

The 7 core models (FR3) are OLS, Ridge, Lasso, Elastic Net, KNN, Random Forest
and Gradient Boosting. SVR (RBF) and Decision Tree are additionally trained
for completeness (FR4 asks for their results to be recorded); they are excluded
from the dashboard picker per the project's hard constraint (AGENTS.md).
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from src.features.engineer import FEATURE_COLUMNS

NAIVE_BASELINE = "naive_baseline"
SPLIT_DATE = "2024-01-01"
RANDOM_STATE = 42
SCALED_MODELS = {"OLS", "Ridge", "Lasso", "Elastic Net", "KNN"}

CORE_MODEL_FACTORIES = {
    "OLS": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=1.0),
    "Lasso": lambda: Lasso(alpha=1.0, max_iter=10000),
    "Elastic Net": lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, max_iter=10000),
    "KNN": lambda: KNeighborsRegressor(n_neighbors=5),
    "Random Forest": lambda: RandomForestRegressor(
        n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
    ),
    "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=RANDOM_STATE),
}

EXTRA_MODEL_FACTORIES = {
    "SVR (RBF)": lambda: Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf"))]),
    "Decision Tree": lambda: DecisionTreeRegressor(random_state=RANDOM_STATE),
}

CORE_MODEL_NAMES = list(CORE_MODEL_FACTORIES)


def model_factories():
    """All 9 model factories: 7 core + 2 completeness models (SVR, Decision Tree)."""
    return {**CORE_MODEL_FACTORIES, **EXTRA_MODEL_FACTORIES}


def make_model(name):
    """Build a fresh (unfitted) model, wrapped in a scaler pipeline when needed."""
    model = model_factories()[name]()
    if name in SCALED_MODELS:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


def chronological_split(features):
    """Chronological train/test split shared by every model and the baseline."""
    X = features[FEATURE_COLUMNS]
    y = features["peak_mw"]
    is_train = features.index < pd.Timestamp(SPLIT_DATE)
    return X[is_train], X[~is_train], y[is_train], y[~is_train]


def train_all(features):
    """Fit all models on the chronological split. Returns {name: fitted model}."""
    X_train, _, y_train, _ = chronological_split(features)
    return {name: make_model(name).fit(X_train, y_train) for name in model_factories()}
