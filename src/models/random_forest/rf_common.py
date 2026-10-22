"""Shared helpers for the random-forest notebooks in this folder (01-05).

Each notebook starts with

    from rf_common import *

which finds the repository root from the notebook's location, puts src/ on the
import path and gives the notebook the team's loader (model_data), the station
reference used by the anomaly models and a few small utilities.

Data:   data/model/model_features_{feb1,mar1,mar15}.csv (read as CSV, no MySQL)
Output: results/rf/<step folder>/
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def find_root(start=None):
    """The sakura_analytics folder that contains this notebook."""
    start = Path(start or Path.cwd()).resolve()
    for folder in [start, *start.parents]:
        if (folder / "src" / "model_data.py").is_file() and \
                (folder / "data" / "model" / "model_features_mar15.csv").is_file():
            return folder
    raise FileNotFoundError(
        "Open the notebook from inside the sakura_analytics repository "
        "(src/models/random_forest/)."
    )


ROOT = find_root()
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from model_data import (  # noqa: E402
    TARGET,
    cv_folds,
    evaluate,
    feature_columns,
    load_features,
)

RF_RESULTS = ROOT / "results" / "rf"

# Forecast dates as they appear in dataset names and file names.
DATES = {"Feb1": "feb1", "Mar1": "mar1", "Mar15": "mar15"}
SAMPLES = ("large", "small")

RANDOM_STATE = 42
REFERENCE_YEARS = 30   # station reference = mean of the last 30 training years
MIN_HISTORY = 10       # fewer records than this -> region mean, then overall mean
METRICS = ["MAE", "RMSE", "R2", "hit_rate_3d"]
PARAM_COLUMNS = ["n_estimators", "max_depth", "min_samples_leaf", "max_features"]

# Small sample = stations with a 2025 record (48 stations).
REGION_ORDER = ["Hokkaido", "Tohoku", "Kanto", "Chubu", "Kinki",
                "Chugoku", "Shikoku", "Kyushu", "Okinawa"]
REGION_ALIASES = {"Kansai": "Kinki"}   # the data say Kansai; the report says Kinki


def output_dir(name, override=None):
    """results/rf/<name>/ (created if missing). `override` redirects test runs."""
    folder = Path(override) if override else RF_RESULTS / name
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def small_station_ids():
    """Station IDs with a full-bloom record in 2025 (the 48-station sample)."""
    mar1 = load_features("mar1", source="csv", verbose=False)
    return set(mar1.loc[mar1["bloom_year"] == 2025, "station_id"])


def load_sample(date, sample, development_only=False):
    """Feature table for one forecast date ('feb1'/'mar1'/'mar15') and sample.

    sample='large' keeps all 82 stations, 'small' the 48 stations seen in 2025.
    Rows are sorted by bloom_year, station_id (as load_features returns them).
    """
    if sample not in SAMPLES:
        raise ValueError(f"sample must be one of {SAMPLES}")
    data = load_features(date, source="csv", verbose=False)
    if sample == "small":
        data = data.loc[data["station_id"].isin(small_station_ids())]
    if development_only:
        data = data.loc[data["split"] == "development"]
    return data.copy()


def dataset_name(date, sample):
    """'mar15', 'small' -> 'Mar15-small'."""
    label = {value: key for key, value in DATES.items()}[date]
    return f"{label}-{sample}"


# ----------------------------------------------------------------------------
# Station reference (anomaly target)
# ----------------------------------------------------------------------------
def make_reference(train, years=REFERENCE_YEARS):
    """Station mean full-bloom DOY over the last `years` training seasons."""
    end = int(train["bloom_year"].max())
    start = end - years + 1
    history = train.loc[train["bloom_year"].between(start, end)]
    return {
        "station": history.groupby("station_id")[TARGET].agg(["mean", "size"]),
        "region": history.groupby("region")[TARGET].mean(),
        "overall": float(history[TARGET].mean()),
        "start": start,
        "end": end,
    }


def reference_values(reference, rows, min_history=MIN_HISTORY):
    """Reference DOY for each row: station mean, else region mean, else overall."""
    means = rows["station_id"].map(reference["station"]["mean"])
    counts = rows["station_id"].map(reference["station"]["size"]).fillna(0)
    fallback = rows["region"].map(reference["region"]).fillna(reference["overall"])
    return np.where(counts >= min_history, means, fallback).astype(float)


def fit_anomaly_rf(train, features, params, reference=None, n_jobs=2):
    """Random forest on (DOY - station reference). Returns (model, reference)."""
    if reference is None:
        reference = make_reference(train)
    model = RandomForestRegressor(**params, random_state=RANDOM_STATE, n_jobs=n_jobs)
    model.fit(train[features], train[TARGET].to_numpy() - reference_values(reference, train))
    return model, reference


def predict_doy(model, reference, rows, features):
    """Predicted anomaly + station reference = predicted full-bloom DOY."""
    return model.predict(rows[features]) + reference_values(reference, rows)


# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def parse_params(row):
    """Hyperparameters from a results-CSV row, with the original Python types."""
    depth = row["max_depth"]
    depth = None if pd.isna(depth) or str(depth).lower() in ("none", "nan") else int(float(depth))
    max_features = str(row["max_features"])
    if max_features not in ("sqrt", "log2"):
        max_features = float(max_features)
        if max_features > 1 and max_features.is_integer():
            max_features = int(max_features)
    return {
        "n_estimators": int(row["n_estimators"]),
        "max_depth": depth,
        "min_samples_leaf": int(row["min_samples_leaf"]),
        "max_features": max_features,
    }


def best_row(grid):
    """Lowest mean validation MAE (ties -> lowest candidate_id)."""
    return grid.sort_values(["valid_MAE", "candidate_id"], kind="stable").iloc[0]


def check_folds(train, valid, label=""):
    if train.empty or valid.empty:
        raise ValueError(f"{label}: empty fold")
    if train["bloom_year"].max() >= valid["bloom_year"].min():
        raise ValueError(f"{label}: training years overlap validation years")
    if valid["bloom_year"].max() > 2014:
        raise ValueError(f"{label}: validation reaches the test years")


def canonical_region(region):
    return REGION_ALIASES.get(region, region)


def save_csv(table, path, index=False):
    """Write via a temporary file so an interrupted run never leaves half a CSV."""
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    table.to_csv(temporary, index=index, encoding="utf-8-sig")
    temporary.replace(path)
    return path


def read_csv(path, **kwargs):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing result file - run the earlier notebook first:\n{path}")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                          encoding="utf-8")


__all__ = [
    "ROOT", "RF_RESULTS", "TARGET", "DATES", "SAMPLES", "RANDOM_STATE",
    "REFERENCE_YEARS", "MIN_HISTORY", "METRICS", "PARAM_COLUMNS", "REGION_ORDER",
    "load_features", "feature_columns", "cv_folds", "evaluate",
    "output_dir", "small_station_ids", "load_sample", "dataset_name",
    "make_reference", "reference_values", "fit_anomaly_rf", "predict_doy",
    "parse_params", "best_row", "check_folds", "canonical_region",
    "save_csv", "read_csv", "save_json", "np", "pd",
]
