"""SQLite read/write helpers for db/forecaster.db (see architecture.md)."""

import math
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "db" / "forecaster.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS features (
    date TEXT PRIMARY KEY,
    peak_mw INTEGER,
    shortage_mw INTEGER,
    energy_met_mu INTEGER,
    energy_shortage_mu REAL,
    Delhi_TMax REAL, Delhi_TMin REAL,
    Bengaluru_TMax REAL, Bengaluru_TMin REAL,
    Chennai_TMax REAL, Chennai_TMin REAL,
    Hyderabad_TMax REAL, Hyderabad_TMin REAL,
    Kolkata_TMax REAL, Kolkata_TMin REAL,
    Mumbai_TMax REAL, Mumbai_TMin REAL,
    Pune_TMax REAL, Pune_TMin REAL,
    Natl_TMax REAL, Natl_TMin REAL,
    DayOfYear INTEGER, Month INTEGER, DOW INTEGER, IsWeekend INTEGER,
    peak_mw_lag1 REAL,
    consecutive_hot_days INTEGER,
    energy_shortage_ratio REAL
);
CREATE TABLE IF NOT EXISTS predictions (
    date TEXT,
    model_name TEXT,
    actual_peak_mw REAL,
    predicted_peak_mw REAL
);
CREATE TABLE IF NOT EXISTS metrics (
    model_name TEXT PRIMARY KEY,
    mae REAL,
    rmse REAL,
    test_r2 REAL,
    cv_r2 REAL,
    qualifies_for_dashboard INTEGER
);
"""


def connect(db_path=DB_PATH):
    """Open (creating if needed) the SQLite database and its tables."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def _write_table(conn, table, df):
    """Replace a table's contents; df.index must be the table's first column.

    The INSERT names its columns explicitly so a mismatch between the
    DataFrame's column order and the CREATE TABLE order can never silently
    swap values.
    """
    conn.execute(f"DELETE FROM {table}")
    columns = []
    for name in df.columns:
        col = df[name].tolist()  # native Python types, as sqlite3 requires
        columns.append([
            None if isinstance(v, float) and math.isnan(v) else v for v in col
        ])
    if isinstance(df.index, pd.DatetimeIndex):
        keys = [str(d.date()) for d in df.index]
    else:
        keys = [str(k) for k in df.index]
    col_names = [df.index.name or "date"] + list(df.columns)
    names_sql = ", ".join(f'"{c}"' for c in col_names)
    placeholders = ", ".join(["?"] * len(col_names))
    conn.executemany(
        f"INSERT INTO {table} ({names_sql}) VALUES ({placeholders})", zip(keys, *columns)
    )
    conn.commit()


def write_features(conn, df):
    """Replace the features table with the engineered feature table (indexed by date)."""
    _write_table(conn, "features", df)


def read_features(conn):
    """Read the features table as a date-indexed DataFrame."""
    df = pd.read_sql("SELECT * FROM features", conn)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def write_predictions(conn, df):
    """Replace the predictions table (df indexed by date)."""
    _write_table(conn, "predictions", df)


def read_predictions(conn, model_name=None):
    """Read per-model predictions, optionally filtered to one model."""
    query, params = "SELECT * FROM predictions", ()
    if model_name is not None:
        query, params = "SELECT * FROM predictions WHERE model_name = ?", (model_name,)
    df = pd.read_sql(query, conn, params=params)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


def write_metrics(conn, df):
    """Replace the metrics table (df indexed by model_name)."""
    _write_table(conn, "metrics", df)


def read_metrics(conn):
    """Read the metrics table ranked by test R2 (best first)."""
    df = pd.read_sql("SELECT * FROM metrics", conn)
    return df.sort_values("test_r2", ascending=False).reset_index(drop=True)
