"""Model registry: save/load trained artifacts and dashboard qualification.

Qualification rule (FR3): a model appears in the dashboard's model picker if
its test R2 > 0.5. Additionally, per the project's hard constraint (AGENTS.md),
SVR (RBF) and Decision Tree never appear in the picker; their metrics are still
reported for completeness (FR4).
"""

from pathlib import Path

import joblib

from src.models.train import NAIVE_BASELINE

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
R2_THRESHOLD = 0.5
EXCLUDED_FROM_PICKER = {"SVR (RBF)", "Decision Tree"}


def qualifies_for_dashboard(model_name, test_r2):
    """True if the model may appear in the dashboard's model picker.

    Rule (FR3): test R2 > 0.5. SVR (RBF) and Decision Tree are additionally
    excluded per the project's hard constraint (AGENTS.md). The naive baseline
    is a benchmark, always shown separately in the dashboard, never selectable.
    """
    return (
        test_r2 > R2_THRESHOLD
        and model_name not in EXCLUDED_FROM_PICKER
        and model_name != NAIVE_BASELINE
    )


def model_slug(name):
    """Filesystem-safe artifact name, e.g. 'Random Forest' -> 'random_forest'."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def save_model(model, name):
    """Persist a fitted model artifact. Returns the artifact path."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / f"{model_slug(name)}.joblib"
    joblib.dump(model, path)
    return path


def load_model(name):
    """Load a previously saved model artifact."""
    return joblib.load(MODELS_DIR / f"{model_slug(name)}.joblib")


def flag_metrics_rows(metrics_df):
    """Add the qualifies_for_dashboard column to a metrics DataFrame."""
    out = metrics_df.copy()
    out["qualifies_for_dashboard"] = [
        int(qualifies_for_dashboard(name, r2))
        for name, r2 in zip(out["model_name"], out["test_r2"])
    ]
    return out
