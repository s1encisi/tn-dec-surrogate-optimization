from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


DENOISE_FOLDS = [
    {
        "fold": "F1",
        "train_start": "2023-01-01",
        "train_end": "2024-05-31",
        "validation_start": "2024-06-01",
        "validation_end": "2024-08-29",
    },
    {
        "fold": "F2",
        "train_start": "2023-01-01",
        "train_end": "2024-08-29",
        "validation_start": "2024-08-30",
        "validation_end": "2024-11-27",
    },
    {
        "fold": "F3",
        "train_start": "2023-01-01",
        "train_end": "2024-11-27",
        "validation_start": "2024-11-28",
        "validation_end": "2025-02-25",
    },
    {
        "fold": "F4",
        "train_start": "2023-01-01",
        "train_end": "2025-02-25",
        "validation_start": "2025-02-26",
        "validation_end": "2025-05-26",
    },
]

DENOISE_EXOGENOUS = ["Q", "COD", "TN", "NH3N", "T", "PPA", "DO", "MLSS"]
DENOISE_VARIANTS = ["D0_raw", "D1_ema07", "D2_median3"]


def continuous_valid_segments(
    dates: pd.Series,
    normal: pd.Series,
    observed: pd.Series,
) -> pd.Series:
    valid = normal.fillna(False).astype(bool) & observed.fillna(False).astype(bool)
    date_break = pd.to_datetime(dates).diff().ne(pd.Timedelta(days=1))
    new_segment = date_break | ~valid | ~valid.shift(fill_value=False)
    segment = new_segment.cumsum().astype(int)
    return segment.where(valid, -1)


def causal_label_transform(
    dates: pd.Series,
    values: pd.Series,
    normal: pd.Series,
    variant: str,
    alpha: float = 0.7,
) -> pd.Series:
    if variant not in DENOISE_VARIANTS:
        raise ValueError(f"Unknown denoise variant: {variant}")
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = normal.fillna(False).astype(bool) & values.notna()
    segments = continuous_valid_segments(dates, normal, values.notna())
    if variant == "D0_raw":
        result.loc[valid] = values.loc[valid].astype(float)
        return result
    for segment_id in sorted(set(segments.loc[segments >= 0])):
        index = segments.index[segments.eq(segment_id)]
        segment_values = values.loc[index].astype(float)
        if variant == "D1_ema07":
            state: float | None = None
            for row_index, raw_value in segment_values.items():
                state = raw_value if state is None else alpha * raw_value + (1.0 - alpha) * state
                result.loc[row_index] = state
        else:
            result.loc[index] = segment_values.rolling(window=3, min_periods=1).median().to_numpy()
    return result


def build_denoise_next_day_frame(
    frame: pd.DataFrame,
    target: str,
) -> tuple[pd.DataFrame, list[str]]:
    data = frame.sort_values("Date").reset_index(drop=True).copy()
    target_date = pd.to_datetime(data["Date"])
    features = pd.DataFrame({"Date": target_date, "actual": data[target].astype(float)})
    day_of_year = target_date.dt.dayofyear.astype(float)
    features["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.2425)
    features["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.2425)
    feature_names = ["doy_sin", "doy_cos"]
    for column in DENOISE_EXOGENOUS:
        name = f"lag1_{column}"
        features[name] = data[column].shift(1)
        feature_names.append(name)
    target_lags = [1, 2, 7] if target == "TN_out" else [1, 7]
    for lag in target_lags:
        name = f"lag{lag}_{target}"
        features[name] = data[target].shift(lag)
        feature_names.append(name)

    segment = continuous_valid_segments(
        target_date,
        data["is_normal_operation"],
        data[target].notna(),
    )
    max_lag = max(target_lags)
    eligible = (
        data["is_normal_operation"].fillna(False).astype(bool)
        & data[target].notna()
        & segment.ge(0)
        & segment.eq(segment.shift(max_lag))
    )
    features["eligible"] = eligible
    features["study_partition"] = data["study_partition"]
    return features, feature_names


def model_factories(seed: int) -> dict[str, Any]:
    return {
        "Ridge": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=10.0)),
            ]
        ),
        "ExtraTrees": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_features=0.9,
                        min_samples_leaf=3,
                        random_state=seed,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "HistGradientBoosting": lambda: Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=350,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=1.0,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    residual = actual - predicted
    sse = float(np.square(residual).sum())
    centered = actual - float(actual.mean())
    sst = float(np.square(centered).sum())
    nonzero = actual != 0
    return {
        "R2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "RMSE": float(np.sqrt(np.mean(np.square(residual)))),
        "MAE": float(np.mean(np.abs(residual))),
        "MAPE_pct": float(np.mean(np.abs(residual[nonzero] / actual[nonzero])) * 100.0)
        if nonzero.any()
        else np.nan,
        "n": int(len(actual)),
        "n_mape": int(nonzero.sum()),
    }


def block_resample_indices(length: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot resample an empty fold.")
    block = min(block_length, length)
    blocks_needed = int(np.ceil(length / block))
    starts = rng.integers(0, length - block + 1, size=blocks_needed)
    indices = np.concatenate([np.arange(start, start + block) for start in starts])
    return indices[:length]


def paired_block_bootstrap(
    predictions: pd.DataFrame,
    candidate: str,
    n_bootstrap: int,
    block_length: int,
    seed: int,
) -> dict[str, float]:
    raw = predictions.loc[predictions["denoise_variant"].eq("D0_raw")].copy()
    altered = predictions.loc[predictions["denoise_variant"].eq(candidate)].copy()
    keys = ["fold", "Date"]
    paired = raw.merge(
        altered,
        on=keys,
        suffixes=("_raw", "_candidate"),
        validate="one_to_one",
    )
    if not np.allclose(paired["actual_raw"], paired["actual_candidate"]):
        raise ValueError("Raw and candidate variants do not share identical validation truth.")
    rng = np.random.default_rng(seed)
    delta_rmse = np.empty(n_bootstrap, dtype=float)
    delta_mae = np.empty(n_bootstrap, dtype=float)
    grouped = [group.reset_index(drop=True) for _, group in paired.groupby("fold", sort=True)]
    for iteration in range(n_bootstrap):
        sampled = []
        for group in grouped:
            index = block_resample_indices(len(group), block_length, rng)
            sampled.append(group.iloc[index])
        sample = pd.concat(sampled, ignore_index=True)
        actual = sample["actual_raw"].to_numpy(float)
        raw_error = actual - sample["prediction_raw"].to_numpy(float)
        candidate_error = actual - sample["prediction_candidate"].to_numpy(float)
        delta_rmse[iteration] = np.sqrt(np.mean(raw_error**2)) - np.sqrt(
            np.mean(candidate_error**2)
        )
        delta_mae[iteration] = np.mean(np.abs(raw_error)) - np.mean(np.abs(candidate_error))

    actual = paired["actual_raw"].to_numpy(float)
    raw_error = actual - paired["prediction_raw"].to_numpy(float)
    candidate_error = actual - paired["prediction_candidate"].to_numpy(float)
    observed_delta_rmse = float(
        np.sqrt(np.mean(raw_error**2)) - np.sqrt(np.mean(candidate_error**2))
    )
    observed_delta_mae = float(np.mean(np.abs(raw_error)) - np.mean(np.abs(candidate_error)))
    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))
    return {
        "delta_RMSE": observed_delta_rmse,
        "relative_RMSE_improvement_pct": 100.0 * observed_delta_rmse / raw_rmse,
        "delta_RMSE_ci_low": float(np.quantile(delta_rmse, 0.025)),
        "delta_RMSE_ci_high": float(np.quantile(delta_rmse, 0.975)),
        "delta_MAE": observed_delta_mae,
        "delta_MAE_ci_low": float(np.quantile(delta_mae, 0.025)),
        "delta_MAE_ci_high": float(np.quantile(delta_mae, 0.975)),
        "bootstrap_probability_improved": float(np.mean(delta_rmse > 0)),
        "n_bootstrap": int(n_bootstrap),
        "block_length": int(block_length),
        "n_oof": int(len(paired)),
    }


def label_preservation_metrics(
    dates: pd.Series,
    raw: pd.Series,
    altered: pd.Series,
) -> dict[str, float | int]:
    valid = raw.notna() & altered.notna()
    raw_values = raw.loc[valid].to_numpy(float)
    altered_values = altered.loc[valid].to_numpy(float)
    variance_ratio = float(np.var(altered_values, ddof=1) / np.var(raw_values, ddof=1))
    raw_diff = np.diff(raw_values)
    altered_diff = np.diff(altered_values)
    diff_variance_ratio = float(
        np.var(altered_diff, ddof=1) / np.var(raw_diff, ddof=1)
    )
    raw_median = float(np.median(raw_values))
    raw_q90 = float(np.quantile(raw_values, 0.90))
    altered_q90 = float(np.quantile(altered_values, 0.90))
    high = raw_values >= raw_q90
    denominator = float(np.sum(raw_values[high] - raw_median))
    peak_amplitude_ratio = (
        float(np.sum(altered_values[high] - raw_median) / denominator)
        if denominator != 0
        else np.nan
    )

    local_peak_indices = [
        index
        for index in range(1, len(raw_values) - 1)
        if raw_values[index] >= raw_values[index - 1]
        and raw_values[index] > raw_values[index + 1]
        and raw_values[index] >= raw_q90
    ]
    lags: list[int] = []
    for index in local_peak_indices:
        window_start = max(0, index - 1)
        window_end = min(len(altered_values), index + 2)
        local = altered_values[window_start:window_end]
        offset = int(np.argmax(local))
        candidate_index = window_start + offset
        if altered_values[candidate_index] >= altered_q90:
            lags.append(candidate_index - index)
    match_rate = float(len(lags) / len(local_peak_indices)) if local_peak_indices else np.nan
    return {
        "n": int(valid.sum()),
        "variance_ratio": variance_ratio,
        "first_difference_variance_ratio": diff_variance_ratio,
        "peak_amplitude_ratio": peak_amplitude_ratio,
        "n_raw_local_peaks": int(len(local_peak_indices)),
        "peak_match_rate": match_rate,
        "median_signed_lag_days": float(np.median(lags)) if lags else np.nan,
        "q90_absolute_lag_days": float(np.quantile(np.abs(lags), 0.90)) if lags else np.nan,
        "date_start": pd.to_datetime(dates.loc[valid]).min().date().isoformat(),
        "date_end": pd.to_datetime(dates.loc[valid]).max().date().isoformat(),
    }
