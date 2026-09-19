"""Clean the raw data and run the data-quality checks from data-schema.md.

Policies (documented here and in the evaluation report; nothing is fixed silently):
- Duplicate dates: keep the first occurrence.
- Temperature readings outside [0, 50] deg C are invalid: set to NaN, then
  linearly interpolated over time. Weather is a model feature, not the demand
  target, so interpolation is acceptable; demand is never interpolated.
- Rows with a missing or non-positive peak_mw / energy_met_mu are dropped:
  demand is the prediction target and cannot be imputed.
- Calendar gaps (missing dates) are reported, never filled.
- Rows on different dates with identical demand+energy values are flagged as
  suspected copy-paste duplicates but kept (no basis to alter them).
"""

from dataclasses import dataclass, field

import pandas as pd

from src.data.load import EXPECTED_COLUMNS

TEMP_BOUNDS = (0.0, 50.0)
TEMPERATURE_COLUMNS = [
    c for c in EXPECTED_COLUMNS if c.endswith("_TMax") or c.endswith("_TMin")
]
DEMAND_COLUMNS = ["peak_mw", "energy_met_mu"]


@dataclass
class CleaningReport:
    """Summary of everything the cleaner found and did."""

    n_rows_raw: int = 0
    n_duplicate_dates: int = 0
    duplicate_dates: list = field(default_factory=list)
    invalid_temp_dates: dict = field(default_factory=dict)
    n_temp_cells_interpolated: int = 0
    dropped_demand_dates: list = field(default_factory=list)
    n_negative_shortage: int = 0
    shortage_filled_dates: dict = field(default_factory=dict)
    suspected_duplicate_demand_dates: list = field(default_factory=list)
    n_date_gaps: int = 0
    date_gaps: list = field(default_factory=list)
    n_rows_clean: int = 0
    date_range: tuple = ()


def clean(df):
    """Apply all cleaning policies. Returns (cleaned_df, CleaningReport)."""
    report = CleaningReport(n_rows_raw=len(df))

    # Duplicate dates: keep the first occurrence.
    dup_mask = df.index.duplicated(keep="first")
    report.duplicate_dates = [str(d.date()) for d in df.index[dup_mask]]
    report.n_duplicate_dates = int(dup_mask.sum())
    df = df[~dup_mask]

    # Temperatures outside plausible bounds are invalid readings.
    lo, hi = TEMP_BOUNDS
    for col in TEMPERATURE_COLUMNS:
        bad = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
        if bad.any():
            report.invalid_temp_dates[col] = [str(d.date()) for d in df.index[bad]]
            df.loc[bad, col] = float("nan")
    n_null_before = int(df[TEMPERATURE_COLUMNS].isna().sum().sum())
    df[TEMPERATURE_COLUMNS] = df[TEMPERATURE_COLUMNS].interpolate(
        method="time", limit_direction="both"
    )
    n_null_after = int(df[TEMPERATURE_COLUMNS].isna().sum().sum())
    report.n_temp_cells_interpolated = n_null_before - n_null_after

    # Demand/energy rows failing the presence/positivity checks are dropped.
    bad_demand = df[DEMAND_COLUMNS].isna().any(axis=1) | (df[DEMAND_COLUMNS] <= 0).any(axis=1)
    report.dropped_demand_dates = [str(d.date()) for d in df.index[bad_demand]]
    df = df[~bad_demand]

    # shortage columns must be >= 0 (check only).
    report.n_negative_shortage = int(
        (df[["shortage_mw", "energy_shortage_mu"]] < 0).sum().sum()
    )

    # A missing shortage reading means no shortage was reported that day: fill
    # with 0 (documented choice; shortage is not the demand target, and both
    # neighbouring days around each gap report small values).
    for col in ["shortage_mw", "energy_shortage_mu"]:
        missing = df[col].isna()
        if missing.any():
            report.shortage_filled_dates[col] = [str(d.date()) for d in df.index[missing]]
    df[["shortage_mw", "energy_shortage_mu"]] = df[
        ["shortage_mw", "energy_shortage_mu"]
    ].fillna(0)

    # Suspected copy-paste rows: identical demand+energy values on different dates.
    value_cols = ["peak_mw", "shortage_mw", "energy_met_mu", "energy_shortage_mu"]
    dup_values = df.duplicated(subset=value_cols, keep=False)
    report.suspected_duplicate_demand_dates = [str(d.date()) for d in df.index[dup_values]]

    # Calendar gaps: dates missing between the first and last record.
    full_range = pd.date_range(df.index.min(), df.index.max(), freq="D")
    missing = full_range.difference(df.index)
    report.date_gaps = [str(d.date()) for d in missing]
    report.n_date_gaps = len(missing)

    report.n_rows_clean = len(df)
    report.date_range = (str(df.index.min().date()), str(df.index.max().date()))
    return df, report
