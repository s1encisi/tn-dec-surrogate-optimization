"""Strict reader for the frozen paper starting dataset.

The paper workflow starts from a materialized, model-ready daily table.  This
module validates that table and exposes the same ``PairedFeatureSet`` contract
used by the frozen prediction, XAI and surrogate-RL implementations.  It does
not construct, impute or scale predictor values.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .enhanced_random import RandomFeatureBundle
from .io import sha256_file
from .paired_random_windows import FEATURE_KEY, PairedFeatureSet


TARGETS = ("TN_out", "DEC")
TN_FEATURES = (
    "doy_sin",
    "doy_cos",
    "time_index_days",
    "Q",
    "COD",
    "TN_in",
    "NH3N",
    "T",
    "PPA",
    "DO",
    "MLSS",
)
DEC_FEATURES = tuple(feature for feature in TN_FEATURES if feature != "TN_in")
RL_QUEUE_COLUMNS = (
    "decision_date",
    "PPA_older",
    "PPA_recent",
    "PPA_historical_action",
    "DO_older",
    "DO_recent",
    "DO_historical_action",
    "MLSS_current",
)
RL_AUDIT_COLUMNS = ("PPA_current_observed", "DO_current_observed")
EXPECTED_COLUMNS = (
    "Date",
    "study_partition",
    *TARGETS,
    *TN_FEATURES,
    *RL_QUEUE_COLUMNS,
    *RL_AUDIT_COLUMNS,
)
EXPECTED_ROWS = 1_076
EXPECTED_YEAR_COUNTS = {2023: 362, 2024: 366, 2025: 348}
EXPECTED_PARTITION_COUNTS = {"development": 906, "fixed_test": 167, "future_unused": 3}
EXPECTED_DO_MISSING_DATES = pd.DatetimeIndex(
    pd.to_datetime(["2024-03-01", "2024-03-02", "2024-03-03"])
)


class InitialDatasetError(ValueError):
    """Raised when the frozen initial-dataset contract is violated."""


@dataclass(frozen=True)
class InitialDataset:
    """Validated starting table plus model-facing feature bundles."""

    path: Path
    file_sha256: str
    manifest_path: Path
    manifest: Mapping[str, Any]
    frame: pd.DataFrame
    feature_set: PairedFeatureSet

    @property
    def source_sha256(self) -> str:
        return str(self.manifest["source_sha256"])


def _read_manifest(path: Path, file_hash: str) -> dict[str, Any]:
    manifest_path = path.with_name("initial_dataset_manifest.json")
    if not manifest_path.is_file():
        raise InitialDatasetError(f"Missing initial-dataset manifest: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InitialDatasetError("The initial-dataset manifest is unreadable.") from exc
    if not isinstance(manifest, dict) or manifest.get("status") != "FROZEN":
        raise InitialDatasetError("The initial-dataset manifest is not frozen.")
    if str(manifest.get("output_sha256", "")).lower() != file_hash.lower():
        raise InitialDatasetError("The initial-dataset file hash differs from its manifest.")
    return manifest


def _validate_schema(frame: pd.DataFrame, manifest: Mapping[str, Any]) -> pd.DataFrame:
    if tuple(manifest.get("tn_predictors", ())) != TN_FEATURES:
        raise InitialDatasetError("Manifest TN predictors differ from the frozen contract.")
    if tuple(manifest.get("dec_predictors", ())) != DEC_FEATURES:
        raise InitialDatasetError("Manifest DEC predictors differ from the frozen contract.")
    if tuple(manifest.get("rl_audit_only_columns", ())) != RL_AUDIT_COLUMNS:
        raise InitialDatasetError("Manifest RL audit fields differ from the frozen contract.")
    if manifest.get("feature_version") != FEATURE_KEY:
        raise InitialDatasetError("Manifest feature version differs from the frozen contract.")
    observed_columns = tuple(str(column) for column in frame.columns)
    if observed_columns != EXPECTED_COLUMNS:
        missing = sorted(set(EXPECTED_COLUMNS).difference(observed_columns))
        extra = sorted(set(observed_columns).difference(EXPECTED_COLUMNS))
        raise InitialDatasetError(
            "Initial-dataset columns or order differ from the frozen contract; "
            f"missing={missing}, extra={extra}."
        )
    if len(frame) != EXPECTED_ROWS or int(manifest.get("rows", -1)) != EXPECTED_ROWS:
        raise InitialDatasetError(f"Expected {EXPECTED_ROWS} rows, observed {len(frame)}.")
    if int(manifest.get("columns", -1)) != len(EXPECTED_COLUMNS):
        raise InitialDatasetError("Manifest column count differs from the frozen contract.")

    data = frame.copy()
    for column in ("Date", "decision_date"):
        data[column] = pd.to_datetime(data[column], errors="raise").dt.normalize()
    if data["Date"].duplicated().any() or not data["Date"].is_monotonic_increasing:
        raise InitialDatasetError("Dates must be unique and strictly ordered.")
    if data["Date"].min() != pd.Timestamp("2023-01-04"):
        raise InitialDatasetError("Unexpected initial-dataset start date.")
    if data["Date"].max() != pd.Timestamp("2025-12-28"):
        raise InitialDatasetError("Unexpected initial-dataset end date.")
    if not data["decision_date"].equals(data["Date"] - pd.Timedelta(days=1)):
        raise InitialDatasetError("Every decision_date must equal Date minus one day.")

    year_counts = data["Date"].dt.year.value_counts().sort_index().to_dict()
    if year_counts != EXPECTED_YEAR_COUNTS:
        raise InitialDatasetError(f"Unexpected year counts: {year_counts}.")
    partition_counts = data["study_partition"].astype(str).value_counts().to_dict()
    if partition_counts != EXPECTED_PARTITION_COUNTS:
        raise InitialDatasetError(f"Unexpected study-partition counts: {partition_counts}.")

    numeric_columns = tuple(
        column
        for column in EXPECTED_COLUMNS
        if column not in {"Date", "decision_date", "study_partition"}
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if np.isinf(data[column].to_numpy(float)).any():
            raise InitialDatasetError(f"{column} contains infinite values.")
    if data.loc[:, list(TARGETS)].isna().any().any():
        raise InitialDatasetError("Prediction targets must remain observed.")

    allowed_missing = {
        "DO": EXPECTED_DO_MISSING_DATES,
        "DO_older": pd.DatetimeIndex([pd.Timestamp("2024-03-03")]),
        "DO_recent": pd.DatetimeIndex([pd.Timestamp("2024-03-02")]),
        "DO_historical_action": pd.DatetimeIndex([pd.Timestamp("2024-03-01")]),
        "DO_current_observed": pd.DatetimeIndex([pd.Timestamp("2024-02-29")]),
    }
    for column in numeric_columns:
        missing_dates = pd.DatetimeIndex(data.loc[data[column].isna(), "Date"])
        expected = allowed_missing.get(column, pd.DatetimeIndex([]))
        if not missing_dates.equals(expected):
            raise InitialDatasetError(
                f"Unexpected missing-value pattern in {column}: "
                f"{missing_dates.strftime('%Y-%m-%d').tolist()}."
            )

    ppa_reconstruction = data[
        ["PPA_older", "PPA_recent", "PPA_historical_action"]
    ].mean(axis=1)
    if not np.allclose(
        ppa_reconstruction.to_numpy(float),
        data["PPA"].to_numpy(float),
        rtol=0.0,
        atol=1e-10,
    ):
        raise InitialDatasetError("PPA queue values do not reproduce the stored model input.")
    do_reconstruction = data[
        ["DO_older", "DO_recent", "DO_historical_action"]
    ].mean(axis=1, skipna=False)
    if not np.allclose(
        do_reconstruction.to_numpy(float),
        data["DO"].to_numpy(float),
        rtol=0.0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise InitialDatasetError("DO queue values do not reproduce the stored model input.")
    return data


def _feature_registry() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, features in (("TN_out", TN_FEATURES), ("DEC", DEC_FEATURES)):
        for feature in features:
            if feature in {"doy_sin", "doy_cos"}:
                source = "Date"
                formula = "sin/cos(2*pi*day_of_year/365.2425)"
                lag = 0
            elif feature == "time_index_days":
                source = "Date"
                formula = "Date - 2023-01-01"
                lag = 0
            else:
                source = "TN" if feature == "TN_in" else feature
                formula = "materialized_model_input"
                lag = 1
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "source_variable": source,
                    "formula": formula,
                    "minimum_lag_days": lag,
                    "target_history": False,
                    "effluent_feature": False,
                }
            )
    return pd.DataFrame.from_records(rows)


def _as_feature_set(frame: pd.DataFrame) -> PairedFeatureSet:
    dates = pd.DatetimeIndex(frame["Date"])
    bundles: dict[str, RandomFeatureBundle] = {}
    for target, features in (("TN_out", TN_FEATURES), ("DEC", DEC_FEATURES)):
        anchor = pd.DataFrame(
            {
                "row_index": np.arange(len(frame), dtype=int),
                "Date": dates,
                "actual": frame[target].to_numpy(float),
                "study_partition": frame["study_partition"].astype(str).to_numpy(),
            }
        )
        bundles[target] = RandomFeatureBundle(
            target=target,
            anchor=anchor,
            matrices={FEATURE_KEY: frame.loc[:, list(features)].copy()},
            feature_names={FEATURE_KEY: tuple(features)},
            dec_history_reset=None,
        )
    return PairedFeatureSet(
        bundles=bundles,
        common_dates=dates,
        feature_registry=_feature_registry(),
    )


def load_initial_dataset(path: str | Path) -> InitialDataset:
    """Load and validate the frozen table without fitting any preprocessing."""

    dataset_path = Path(path).resolve()
    if not dataset_path.is_file():
        raise InitialDatasetError(f"Initial dataset does not exist: {dataset_path}")
    file_hash = sha256_file(dataset_path)
    manifest_path = dataset_path.with_name("initial_dataset_manifest.json")
    manifest = _read_manifest(dataset_path, file_hash)
    frame = pd.read_csv(
        dataset_path,
        encoding="utf-8-sig",
        float_precision="round_trip",
    )
    frame = _validate_schema(frame, manifest)
    return InitialDataset(
        path=dataset_path,
        file_sha256=file_hash,
        manifest_path=manifest_path,
        manifest=manifest,
        frame=frame,
        feature_set=_as_feature_set(frame),
    )
