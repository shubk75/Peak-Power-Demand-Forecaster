"""Integration tests for the modeling layer (src/models/*).

Run from the repo root:  python -m unittest discover -s tests
If db/forecaster.db or the model artifacts are missing, the full pipeline runs
first (it takes well under a minute on the ~1,000-row dataset).

Note: the metrics table holds the 7 core models + the naive baseline (the 8
rows required by tasks.md) plus 2 completeness models (SVR, Decision Tree)
that requirements.md FR4 asks to evaluate and record; both are excluded from
the dashboard picker, so the flag logic below checks for that too.
"""

import unittest

import numpy as np

import run_pipeline
from src.data.db import DB_PATH, connect, read_metrics
from src.models.registry import (
    EXCLUDED_FROM_PICKER,
    MODELS_DIR,
    load_model,
    qualifies_for_dashboard,
)
from src.models.train import CORE_MODEL_NAMES, NAIVE_BASELINE


class TestModelTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DB_PATH.exists() or not (MODELS_DIR / "random_forest.joblib").exists():
            run_pipeline.main()

    def test_all_core_models_trained_without_error(self):
        for name in CORE_MODEL_NAMES:
            model = load_model(name)  # raises if the artifact is missing/corrupt
            self.assertTrue(hasattr(model, "predict"))

    def test_metrics_table_has_required_rows(self):
        conn = connect(DB_PATH)
        metrics = read_metrics(conn)
        conn.close()
        names = set(metrics["model_name"])
        for name in CORE_MODEL_NAMES + [NAIVE_BASELINE]:
            self.assertIn(name, names)
        self.assertGreaterEqual(len(metrics), 8)

    def test_qualifying_flag_logic_matches_r2(self):
        conn = connect(DB_PATH)
        metrics = read_metrics(conn)
        conn.close()
        for _, row in metrics.iterrows():
            expected = int(qualifies_for_dashboard(row["model_name"], row["test_r2"]))
            self.assertEqual(int(row["qualifies_for_dashboard"]), expected)

    def test_excluded_models_are_not_selectable(self):
        conn = connect(DB_PATH)
        metrics = read_metrics(conn)
        conn.close()
        for _, row in metrics.iterrows():
            if row["model_name"] in EXCLUDED_FROM_PICKER:
                self.assertEqual(int(row["qualifies_for_dashboard"]), 0)

    def test_naive_baseline_has_finite_metrics(self):
        conn = connect(DB_PATH)
        metrics = read_metrics(conn)
        conn.close()
        naive = metrics[metrics["model_name"] == NAIVE_BASELINE].iloc[0]
        for col in ("mae", "rmse", "test_r2", "cv_r2"):
            self.assertTrue(np.isfinite(naive[col]))


if __name__ == "__main__":
    unittest.main()
