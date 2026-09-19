"""Load and validate the raw dataset against data-schema.md."""

from pathlib import Path

import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "dataset.csv"

EXPECTED_COLUMNS = [
    "date", "peak_mw", "shortage_mw", "energy_met_mu", "energy_shortage_mu",
    "Delhi_TMax", "Delhi_TMin", "Bengaluru_TMax", "Bengaluru_TMin",
    "Chennai_TMax", "Chennai_TMin", "Hyderabad_TMax", "Hyderabad_TMin",
    "Kolkata_TMax", "Kolkata_TMin", "Mumbai_TMax", "Mumbai_TMin",
    "Pune_TMax", "Pune_TMin", "Natl_TMax", "Natl_TMin",
    "DayOfYear", "Month", "DOW", "IsWeekend",
]

INTEGER_COLUMNS = [
    "peak_mw", "shortage_mw", "energy_met_mu",
    "DayOfYear", "Month", "DOW", "IsWeekend",
]
FLOAT_COLUMNS = ["energy_shortage_mu"] + [
    c for c in EXPECTED_COLUMNS if c.endswith("_TMax") or c.endswith("_TMin")
]


def load_raw(path=RAW_PATH):
    """Read dataset.csv, validate its schema, and return a date-indexed DataFrame."""
    df = pd.read_csv(path)
    validate_schema(df)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    return df.set_index("date").sort_index()


def validate_schema(df):
    """Raise ValueError if columns or values do not match data-schema.md."""
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"dataset.csv is missing expected columns: {missing}")
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if extra:
        raise ValueError(f"dataset.csv has unexpected columns: {extra}")
    for col in INTEGER_COLUMNS + FLOAT_COLUMNS:
        coerced = pd.to_numeric(df[col], errors="coerce")
        n_new_nulls = int(coerced.isna().sum()) - int(df[col].isna().sum())
        if n_new_nulls > 0:
            raise ValueError(f"column {col!r} has {n_new_nulls} non-numeric values")
