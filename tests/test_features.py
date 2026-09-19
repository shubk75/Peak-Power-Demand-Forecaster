"""Unit tests for src/features/engineer.py.

Run from the repo root:  python -m unittest discover -s tests
"""

import unittest

import numpy as np
import pandas as pd

from src.features.engineer import (
    add_consecutive_hot_days,
    add_energy_shortage_ratio,
    add_peak_mw_lag1,
    compute_hot_threshold,
)


def mini_df(temps, demand=None, energy_met_mu=None, shortage_mu=None):
    """Build a small daily DataFrame with a Natl_TMax column for testing."""
    n = len(temps)
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Natl_TMax": temps,
            "peak_mw": demand or list(range(100, 100 * (n + 1), 100)),
            "energy_met_mu": energy_met_mu or [1000] * n,
            "energy_shortage_mu": shortage_mu or [10.0] * n,
        },
        index=dates,
    )


class TestConsecutiveHotDays(unittest.TestCase):
    def test_known_runs(self):
        """Hand-built series with known runs of hot days (threshold 38)."""
        df = add_consecutive_hot_days(mini_df([37, 39, 40, 35, 41, 42, 38]), threshold=38.0)
        self.assertEqual(df["consecutive_hot_days"].tolist(), [0, 1, 2, 0, 1, 2, 0])

    def test_threshold_is_strictly_greater(self):
        """A day exactly at the threshold is not hot."""
        df = add_consecutive_hot_days(mini_df([38, 38, 39]), threshold=38.0)
        self.assertEqual(df["consecutive_hot_days"].tolist(), [0, 0, 1])

    def test_leading_hot_days_count_from_start(self):
        df = add_consecutive_hot_days(mini_df([40, 41, 30, 42]), threshold=38.0)
        self.assertEqual(df["consecutive_hot_days"].tolist(), [1, 2, 0, 1])


class TestPeakMwLag1(unittest.TestCase):
    def test_alignment_and_first_row_null(self):
        df = add_peak_mw_lag1(mini_df([30, 31, 32], demand=[100, 200, 300]))
        self.assertTrue(np.isnan(df["peak_mw_lag1"].iloc[0]))
        self.assertEqual(df["peak_mw_lag1"].iloc[1], 100)
        self.assertEqual(df["peak_mw_lag1"].iloc[2], 200)

    def test_lag_refers_to_previous_available_row_after_gap(self):
        """After a calendar gap the lag is the last observed day (no interpolation)."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-03"])  # 2023-01-02 missing
        df = pd.DataFrame(
            {
                "Natl_TMax": [30, 31],
                "peak_mw": [100, 200],
                "energy_met_mu": [1000, 1000],
                "energy_shortage_mu": [1.0, 1.0],
            },
            index=dates,
        )
        df = add_peak_mw_lag1(df)
        self.assertEqual(df["peak_mw_lag1"].iloc[1], 100)


class TestEnergyShortageRatio(unittest.TestCase):
    def test_ratio(self):
        df = add_energy_shortage_ratio(mini_df([30], energy_met_mu=[500], shortage_mu=[25.0]))
        self.assertAlmostEqual(df["energy_shortage_ratio"].iloc[0], 0.05)


class TestHotThreshold(unittest.TestCase):
    def test_percentile(self):
        temps = list(range(20, 40))
        self.assertAlmostEqual(
            compute_hot_threshold(mini_df(temps)), float(np.percentile(temps, 90))
        )


if __name__ == "__main__":
    unittest.main()
