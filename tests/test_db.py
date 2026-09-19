"""Round-trip tests for src/data/db.py.

Run from the repo root:  python -m unittest discover -s tests
These catch column-alignment bugs: whatever is written must read back
identically, column for column.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.db import connect, read_features, read_metrics, write_features, write_metrics


def sample_features():
    """A small feature table with distinct values in every column."""
    dates = pd.date_range("2023-01-01", periods=4, freq="D")
    return pd.DataFrame(
        {
            "peak_mw": [100, 200, 300, 400],
            "Natl_TMax": [30.5, 31.5, 39.5, 40.5],
            "peak_mw_lag1": [np.nan, 100.0, 200.0, 300.0],
            "consecutive_hot_days": [0, 0, 1, 2],
            "energy_shortage_ratio": [0.001, 0.002, 0.003, 0.004],
        },
        index=dates,
    )


class TestDbRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "test.db"

    def tearDown(self):
        self.tmp.cleanup()

    def test_features_round_trip_preserves_every_column(self):
        """Written values must read back identically, column for column."""
        source = sample_features()
        conn = connect(self.db_path)
        write_features(conn, source)
        loaded = read_features(conn)
        conn.close()
        for col in source.columns:
            for i in range(len(source)):
                expected, got = source[col].iloc[i], loaded[col].iloc[i]
                if pd.isna(expected):
                    self.assertTrue(pd.isna(got), f"{col} row {i}: expected null")
                else:
                    self.assertAlmostEqual(float(expected), float(got), places=6,
                                           msg=f"{col} row {i} mismatched")

    def test_consecutive_hot_days_and_ratio_not_swapped(self):
        """Regression test: the streak and shortage ratio must not trade places."""
        source = sample_features()
        conn = connect(self.db_path)
        write_features(conn, source)
        loaded = read_features(conn)
        conn.close()
        # A day with streak 2 must not read back as the shortage ratio (0.004).
        self.assertEqual(int(loaded["consecutive_hot_days"].iloc[3]), 2)
        self.assertAlmostEqual(float(loaded["energy_shortage_ratio"].iloc[3]), 0.004)

    def test_metrics_round_trip(self):
        source = pd.DataFrame(
            {
                "mae": [10.0, 20.0],
                "rmse": [12.0, 25.0],
                "test_r2": [0.9, 0.4],
                "cv_r2": [0.8, 0.3],
                "qualifies_for_dashboard": [1, 0],
            },
            index=pd.Index(["best_model", "worst_model"], name="model_name"),
        )
        conn = connect(self.db_path)
        write_metrics(conn, source)
        loaded = read_metrics(conn)
        conn.close()
        self.assertEqual(loaded["model_name"].tolist(), ["best_model", "worst_model"])
        self.assertEqual(int(loaded["qualifies_for_dashboard"].iloc[0]), 1)
        self.assertAlmostEqual(float(loaded["test_r2"].iloc[0]), 0.9)


if __name__ == "__main__":
    unittest.main()
