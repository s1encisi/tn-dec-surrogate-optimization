"""Utilities for the post-hoc advanced prediction extension.

This module deliberately keeps the two evaluation questions separate:

* strict expanding-window one-step forecasting may use causal target history;
* random interpolation diagnostics use F-A only and therefore contain no target lags.

The original fixed test has already been observed.  Nothing in this module reads files or
authorizes a new independent-test claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from .features import build_tabular_next_day_features
from .metrics import regression_metrics
from .modern_sequence import StrictCausalWindows, build_strict_causal_windows


class AdvancedPredictionError(ValueError):
    """Raised when the advanced extension would violate its registered scope."""


@dataclass(frozen=True)
class RandomModelSpec:
    name: str
    estimator: RegressorMixin
    candidates: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RandomFitResult:
    prediction: np.ndarray
    selected_parameters: dict[str, Any]
    selected_candidate: int
    tuning_trials: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SequencePanel:
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    windows: StrictCausalWindows


def validate_development_only(frame: pd.DataFrame, *, target: str) -> pd.DataFrame:
    """Validate a continuous daily development frame and fail closed on other partitions."""

    required = {"Date", "study_partition", "is_normal_operation", target}
    missing = required.difference(frame.columns)
    if missing:
        raise AdvancedPredictionError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.normalize()
    if data["Date"].duplicated().any() or not data["Date"].is_monotonic_increasing:
        raise AdvancedPredictionError("Dates must be unique and increasing.")
    if len(data) and not pd.DatetimeIndex(data["Date"]).equals(
        pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    ):
        raise AdvancedPredictionError("The development calendar must be gap-free.")
    if set(data["study_partition"].astype(str)) != {"development"}:
        raise AdvancedPredictionError("Advanced development functions reject non-development rows.")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    if np.isinf(data[target].to_numpy(float)).any():
        raise AdvancedPredictionError("Target contains infinite values.")
    return data


def build_advanced_sequence_panel(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    target: str,
    *,
    alignment: str = "L2",
    window: int = 28,
) -> SequencePanel:
    """Build F-A covariate histories plus raw target history for sequence models.

    F-A is used inside every window because the window builder appends historical raw target
    values itself.  This avoids duplicating hand-built lags while preserving exactly the same
    registered covariates as the main study.
    """

    data = validate_development_only(frame, target=target)
    panel, feature_names = build_tabular_next_day_features(
        data, feature_config, target, "F_A", alignment
    )
    if any(name.startswith("lag") or "rolling" in name.lower() for name in feature_names):
        raise AdvancedPredictionError("Sequence covariate channels unexpectedly contain lags.")
    windows = build_strict_causal_windows(
        panel["Date"],
        panel[list(feature_names)],
        panel["actual"],
        target_dates=panel["Date"],
        window=window,
        normal=data["is_normal_operation"],
        include_target_history=True,
    )
    return SequencePanel(panel, tuple(feature_names), windows)


def build_advanced_random_registry(seed: int) -> dict[str, RandomModelSpec]:
    """Return six additional random-interpolation regressors and small fixed search spaces."""

    imputer = lambda: SimpleImputer(strategy="median", add_indicator=True)  # noqa: E731
    scaled = lambda model: Pipeline(  # noqa: E731
        [("imputer", imputer()), ("scaler", StandardScaler()), ("model", model)]
    )
    unscaled = lambda model: Pipeline(  # noqa: E731
        [("imputer", imputer()), ("model", model)]
    )

    hist = unscaled(
        HistGradientBoostingRegressor(random_state=seed, max_iter=300, early_stopping=True)
    )
    gradient = unscaled(GradientBoostingRegressor(random_state=seed, loss="huber"))
    ada = unscaled(
        AdaBoostRegressor(
            estimator=DecisionTreeRegressor(random_state=seed, min_samples_leaf=5),
            random_state=seed,
            loss="square",
        )
    )
    knn = scaled(KNeighborsRegressor())
    gp = scaled(
        TransformedTargetRegressor(
            regressor=GaussianProcessRegressor(
                normalize_y=False,
                random_state=seed,
                n_restarts_optimizer=0,
            ),
            transformer=StandardScaler(),
        )
    )
    return {
        "HistGradientBoosting": RandomModelSpec(
            "HistGradientBoosting",
            hist,
            tuple(
                {
                    "model__learning_rate": learning_rate,
                    "model__max_leaf_nodes": leaves,
                    "model__l2_regularization": l2,
                }
                for learning_rate, leaves, l2 in (
                    (0.03, 7, 0.0),
                    (0.03, 15, 1.0),
                    (0.06, 7, 1.0),
                    (0.06, 15, 3.0),
                )
            ),
        ),
        "GradientBoosting": RandomModelSpec(
            "GradientBoosting",
            gradient,
            tuple(
                {
                    "model__n_estimators": estimators,
                    "model__learning_rate": learning_rate,
                    "model__max_depth": depth,
                    "model__min_samples_leaf": leaf,
                }
                for estimators, learning_rate, depth, leaf in (
                    (150, 0.03, 1, 5),
                    (250, 0.03, 2, 8),
                    (150, 0.06, 1, 8),
                    (250, 0.06, 2, 12),
                )
            ),
        ),
        "AdaBoost": RandomModelSpec(
            "AdaBoost",
            ada,
            tuple(
                {
                    "model__n_estimators": estimators,
                    "model__learning_rate": learning_rate,
                    "model__estimator__max_depth": depth,
                }
                for estimators, learning_rate, depth in (
                    (100, 0.03, 2),
                    (250, 0.03, 2),
                    (100, 0.08, 3),
                    (250, 0.08, 3),
                )
            ),
        ),
        "KNN": RandomModelSpec(
            "KNN",
            knn,
            tuple(
                {
                    "model__n_neighbors": neighbors,
                    "model__weights": weights,
                    "model__p": p,
                }
                for neighbors, weights, p in (
                    (5, "distance", 1),
                    (10, "distance", 1),
                    (20, "distance", 2),
                    (30, "uniform", 2),
                )
            ),
        ),
        "GaussianProcess": RandomModelSpec(
            "GaussianProcess",
            gp,
            (
                {
                    "model__regressor__kernel": ConstantKernel(1.0) * RBF(1.0)
                    + WhiteKernel(0.1),
                    "model__regressor__alpha": 1e-4,
                },
                {
                    "model__regressor__kernel": ConstantKernel(1.0) * Matern(1.0, nu=1.5)
                    + WhiteKernel(0.1),
                    "model__regressor__alpha": 1e-3,
                },
            ),
        ),
    }


def fit_random_model_nested(
    spec: RandomModelSpec,
    X: pd.DataFrame,
    y: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    *,
    seed: int,
) -> RandomFitResult:
    """Tune a random-diagnostic model only inside the outer training subset."""

    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if np.intersect1d(train_indices, test_indices).size:
        raise AdvancedPredictionError("Outer random train and test rows overlap.")
    if len(train_indices) < 30 or not len(test_indices):
        raise AdvancedPredictionError("Random diagnostic split is too small.")
    inner_train, inner_validation = train_test_split(
        train_indices, test_size=0.20, random_state=seed + 100_003, shuffle=True
    )
    trials: list[dict[str, Any]] = []
    for candidate_index, parameters in enumerate(spec.candidates):
        try:
            fitted = clone(spec.estimator).set_params(**parameters)
            fitted.fit(X.iloc[inner_train], y[inner_train])
            prediction = np.asarray(fitted.predict(X.iloc[inner_validation]), dtype=float).reshape(-1)
            rmse = float(mean_squared_error(y[inner_validation], prediction) ** 0.5)
            mae = float(mean_absolute_error(y[inner_validation], prediction))
            status = "completed"
            error = None
        except Exception as exc:  # numerical candidate failure must remain auditable
            rmse = np.nan
            mae = np.nan
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        trials.append(
            {
                "model": spec.name,
                "seed": int(seed),
                "candidate_index": candidate_index,
                "parameters": parameters,
                "inner_RMSE": rmse,
                "inner_MAE": mae,
                "status": status,
                "error": error,
                "fixed_test_accessed": False,
            }
        )
    completed = [trial for trial in trials if trial["status"] == "completed"]
    if not completed:
        raise RuntimeError(f"All candidates failed for {spec.name}/seed={seed}.")
    best_rmse = min(float(trial["inner_RMSE"]) for trial in completed)
    near = [
        trial for trial in completed if float(trial["inner_RMSE"]) <= best_rmse * 1.01
    ]
    selected = min(
        near,
        key=lambda trial: (float(trial["inner_MAE"]), int(trial["candidate_index"])),
    )
    fitted = clone(spec.estimator).set_params(**selected["parameters"])
    fitted.fit(X.iloc[train_indices], y[train_indices])
    prediction = np.asarray(fitted.predict(X.iloc[test_indices]), dtype=float).reshape(-1)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"{spec.name} returned non-finite predictions.")
    return RandomFitResult(
        prediction,
        dict(selected["parameters"]),
        int(selected["candidate_index"]),
        tuple(trials),
    )


def metric_table(predictions: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Compute the frozen regression metrics for arbitrary prediction groups."""

    required = {*group_columns, "actual", "prediction"}
    missing = required.difference(predictions.columns)
    if missing:
        raise AdvancedPredictionError(f"Prediction table is missing {sorted(missing)}")
    records: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, observed=True, sort=False):
        normalized_keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, normalized_keys, strict=True))
        record.update(
            regression_metrics(group["actual"].to_numpy(float), group["prediction"].to_numpy(float))
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def random_summary(metrics_by_seed: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated random splits without inventing an aggregate score."""

    metrics = ("R2", "RMSE", "MAE", "MAPE_pct", "WAPE_pct")
    records: list[dict[str, Any]] = []
    for (target, model), group in metrics_by_seed.groupby(
        ["target", "model"], observed=True, sort=False
    ):
        record: dict[str, Any] = {"target": target, "model": model, "n_seeds": len(group)}
        for metric in metrics:
            values = pd.to_numeric(group[metric], errors="raise")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
        records.append(record)
    return pd.DataFrame.from_records(records)
