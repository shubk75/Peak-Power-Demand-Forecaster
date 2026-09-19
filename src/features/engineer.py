"""Feature engineering: lag-1 demand, consecutive hot days, shortage ratio.

Day-ahead framing (the "shift framing" from data-schema.md): the model predicts
peak_mw(t) from previous-day peak demand (peak_mw_lag1), same-day weather and
calendar features, and grid-stress features. Same-day weather stands in for the
day-ahead temperature forecast a grid planner would have. The naive baseline for
this framing predicts peak_mw(t) = peak_mw(t-1).

Derived features
- peak_mw_lag1 = peak_mw(t-1): previous available day's national peak demand.
  The very first row has no previous day, so it is dropped (documented drop
  policy). After a calendar gap the lag refers to the last observed day, which
  may be more than one calendar day earlier; demand is never interpolated.
- consecutive_hot_days(t): running count of consecutive days ending at t where
  Natl_TMax strictly exceeds HOT_TEMP_THRESHOLD (= 90th percentile of Natl_TMax
  over the cleaned dataset, computed by compute_hot_threshold). Resets to 0 on
  any day at or below the threshold.
- energy_shortage_ratio(t) = energy_shortage_mu(t) / energy_met_mu(t): grid
  stress independent of absolute scale (rows with energy_met_mu <= 0 are
  already dropped by clean.py).
"""

import pandas as pd

from src.data.load import EXPECTED_COLUMNS

REFERENCE_SERIES = "Natl_TMax"
HOT_TEMP_PERCENTILE = 0.90

CALENDAR_COLUMNS = ["DayOfYear", "Month", "DOW", "IsWeekend"]
NATIONAL_TEMP_COLUMNS = ["Natl_TMax", "Natl_TMin"]
CITY_TEMP_COLUMNS = [
    c for c in EXPECTED_COLUMNS
    if (c.endswith("_TMax") or c.endswith("_TMin"))
    and c not in NATIONAL_TEMP_COLUMNS
]
TEMPERATURE_COLUMNS = CITY_TEMP_COLUMNS + NATIONAL_TEMP_COLUMNS

FEATURE_COLUMNS = (
    ["peak_mw_lag1", "consecutive_hot_days", "energy_shortage_ratio"]
    + TEMPERATURE_COLUMNS
    + CALENDAR_COLUMNS
)


def compute_hot_threshold(df):
    """Heat threshold: 90th percentile of the reference series (Natl_TMax)."""
    return float(df[REFERENCE_SERIES].quantile(HOT_TEMP_PERCENTILE))


def add_consecutive_hot_days(df, threshold):
    """Return a copy of df with the consecutive_hot_days column recomputed."""
    is_hot = (df[REFERENCE_SERIES] > threshold).astype(int)
    run_id = (is_hot == 0).cumsum()  # new run id on every non-hot day
    out = df.copy()
    out["consecutive_hot_days"] = is_hot.groupby(run_id).cumsum()
    return out


def add_peak_mw_lag1(df):
    """Return a copy of df with the previous-day peak demand column."""
    out = df.copy()
    out["peak_mw_lag1"] = df["peak_mw"].shift(1)
    return out


def add_energy_shortage_ratio(df):
    """Return a copy of df with energy_shortage_mu / energy_met_mu."""
    out = df.copy()
    out["energy_shortage_ratio"] = df["energy_shortage_mu"] / df["energy_met_mu"]
    return out


def build_features(df, hot_threshold=None):
    """Run the full feature pipeline on a cleaned, date-indexed DataFrame.

    Drops the first row (its lag-1 demand is undefined). Raises ValueError if
    any modeling feature still contains nulls, rather than training on NaN.
    """
    if hot_threshold is None:
        hot_threshold = compute_hot_threshold(df)
    out = add_peak_mw_lag1(df)
    out = add_consecutive_hot_days(out, threshold=hot_threshold)
    out = add_energy_shortage_ratio(out)
    out = out.dropna(subset=["peak_mw_lag1"])
    nulls = out[FEATURE_COLUMNS].isna().sum()
    nulls = nulls[nulls > 0]
    if len(nulls):
        raise ValueError(f"engineered features contain nulls: {nulls.to_dict()}")
    return out
