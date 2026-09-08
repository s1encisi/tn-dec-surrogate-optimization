from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import hashlib
import itertools
import re
from typing import Any, Final

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.utils.validation import check_is_fitted
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor


CLASSICAL_MODEL_NAMES: Final[tuple[str, ...]] = (
    "Persistence",
    "SeasonalNaive",
    "AutoReg",
    "SARIMAX",
    "ElasticNet",
    "PLS",
    "SVR",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)

TABULAR_MODEL_NAMES: Final[tuple[str, ...]] = (
    "ElasticNet",
    "PLS",
    "SVR",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
)

SCALED_MODEL_NAMES: Final[frozenset[str]] = frozenset({"ElasticNet", "PLS", "SVR"})
DEFAULT_LEGAL_LAGS: Final[tuple[int, ...]] = (1, 2, 3, 7)
_ALIASES: Final[dict[str, str]] = {"RF": "RandomForest", "SeasonalNaive": "SeasonalNaive"}
_LAG_PATTERN = re.compile(r"^lag(?P<lag>\d+)_(?P<target>.+)$")


@dataclass(frozen=True)
class ModelSpec:
    """A registered estimator and its pre-generated, development-only candidates."""

    name: str
    family: str
    estimator: BaseEstimator | None
    candidates: tuple[dict[str, Any], ...]
    requires_sequential_update: bool = False


def _canonical_name(name: str) -> str:
    canonical = _ALIASES.get(name, name)
    if canonical not in CLASSICAL_MODEL_NAMES:
        raise KeyError(f"Unknown classical model: {name}")
    return canonical


def _validate_candidate_count(candidate_count: int) -> None:
    if isinstance(candidate_count, bool) or not isinstance(candidate_count, int):
        raise TypeError("candidate_count must be an integer.")
    if candidate_count < 1:
        raise ValueError("candidate_count must be at least one.")


def _model_rng(seed: int, model_name: str) -> np.random.Generator:
    digest = hashlib.sha256(model_name.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:8], byteorder="little", signed=False)
    return np.random.default_rng(np.random.SeedSequence([int(seed), offset]))


def _candidate_key(candidate: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key, repr(value)) for key, value in candidate.items()))


def _draw_unique_candidates(
    default: dict[str, Any],
    generator: Callable[[], dict[str, Any]],
    candidate_count: int,
) -> tuple[dict[str, Any], ...]:
    candidates = [default]
    seen = {_candidate_key(default)}
    attempts = 0
    while len(candidates) < candidate_count:
        candidate = generator()
        key = _candidate_key(candidate)
        if key not in seen:
            candidates.append(candidate)
            seen.add(key)
        attempts += 1
        if attempts > candidate_count * 10_000:
            raise RuntimeError("Unable to draw the requested number of unique candidates.")
    return tuple(candidates)


def _lag_columns(
    frame: pd.DataFrame,
    lag_days: Sequence[int],
    target: str | None,
) -> list[str]:
    columns = [str(column) for column in frame.columns]
    selected: list[str] = []
    for lag in lag_days:
        expected = f"lag{lag}_{target}" if target is not None else None
        if expected is not None and expected in columns:
            selected.append(expected)
            continue
        matches = [
            column
            for column in columns
            if (match := _LAG_PATTERN.fullmatch(column)) is not None
            and int(match.group("lag")) == lag
            and (target is None or match.group("target") == target)
        ]
        if not matches:
            suffix = f" for target {target}" if target is not None else ""
            raise ValueError(f"Missing legal lag-{lag} feature{suffix}.")
        if len(matches) > 1:
            raise ValueError(
                f"Lag-{lag} is ambiguous across targets {matches}; pass target explicitly."
            )
        selected.append(matches[0])
    return selected


def _extract_lag_matrix(
    X: Any,
    lag_days: Sequence[int],
    target: str | None,
    lag_indices: Sequence[int] | None,
) -> tuple[np.ndarray, tuple[str, ...] | None, tuple[int, ...] | None]:
    if isinstance(X, pd.DataFrame):
        columns = _lag_columns(X, lag_days, target)
        values = X.loc[:, columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return values, tuple(columns), None

    values = np.asarray(X, dtype=float)
    if values.ndim != 2:
        raise ValueError(f"X must be two-dimensional; observed shape {values.shape}.")
    if lag_indices is None:
        raise ValueError("lag_indices are required when X is not a pandas DataFrame.")
    indices = tuple(int(index) for index in lag_indices)
    if len(indices) != len(tuple(lag_days)):
        raise ValueError("lag_indices and lag_days must have the same length.")
    if any(index < 0 or index >= values.shape[1] for index in indices):
        raise ValueError(f"lag_indices {indices} are invalid for {values.shape[1]} features.")
    return values[:, indices], None, indices


class _LagValueRegressor(RegressorMixin, BaseEstimator):
    def __init__(
        self,
        *,
        lag_day: int,
        target: str | None = None,
        feature_index: int | None = None,
    ) -> None:
        self.lag_day = lag_day
        self.target = target
        self.feature_index = feature_index

    def fit(self, X: Any, y: Any) -> _LagValueRegressor:
        values, columns, indices = _extract_lag_matrix(
            X,
            (self.lag_day,),
            self.target,
            None if self.feature_index is None else (self.feature_index,),
        )
        target_values = np.asarray(y, dtype=float)
        if target_values.ndim != 1 or len(target_values) != len(values):
            raise ValueError("y must be one-dimensional and aligned with X.")
        if not np.isfinite(target_values).all():
            raise ValueError("Training labels must be finite; target imputation is forbidden.")
        self.n_features_in_ = X.shape[1]
        self.lag_columns_ = columns
        self.lag_indices_ = indices
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("n_features_in_", "lag_columns_", "lag_indices_"))
        if isinstance(X, pd.DataFrame) and self.lag_columns_ is not None:
            missing = [column for column in self.lag_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Prediction data are missing fitted lag columns: {missing}")
            values = X.loc[:, list(self.lag_columns_)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
        elif not isinstance(X, pd.DataFrame) and self.lag_indices_ is not None:
            matrix = np.asarray(X, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must have {self.n_features_in_} features; observed shape {matrix.shape}."
                )
            values = matrix[:, list(self.lag_indices_)]
        else:
            raise TypeError("Prediction input type must match the fitted lag representation.")
        if not np.isfinite(values).all():
            raise ValueError("NaN/inf in a required observed target lag cannot be imputed silently.")
        return values[:, 0]


class PersistenceRegressor(_LagValueRegressor):
    """Next-day persistence: predict target date t from the observed value at t-1."""

    def __init__(self, *, target: str | None = None, feature_index: int | None = None) -> None:
        super().__init__(lag_day=1, target=target, feature_index=feature_index)


class SeasonalNaiveRegressor(_LagValueRegressor):
    """Weekly seasonal naive: predict target date t from the observation at t-7."""

    def __init__(self, *, target: str | None = None, feature_index: int | None = None) -> None:
        super().__init__(lag_day=7, target=target, feature_index=feature_index)


class LegalLagAutoRegressor(RegressorMixin, BaseEstimator):
    """OLS AutoReg using only pre-registered, already-observed target lags.

    The imputer and scaler are fitted inside ``fit`` and therefore inside each
    training fold when the estimator is used by a time-series CV runner.
    """

    def __init__(
        self,
        *,
        lag_days: tuple[int, ...] = DEFAULT_LEGAL_LAGS,
        target: str | None = None,
        lag_indices: tuple[int, ...] | None = None,
        fit_intercept: bool = True,
    ) -> None:
        self.lag_days = lag_days
        self.target = target
        self.lag_indices = lag_indices
        self.fit_intercept = fit_intercept

    def fit(self, X: Any, y: Any) -> LegalLagAutoRegressor:
        lag_days = tuple(int(lag) for lag in self.lag_days)
        if not lag_days or any(lag < 1 for lag in lag_days) or len(set(lag_days)) != len(lag_days):
            raise ValueError("lag_days must contain unique positive integers.")
        values, columns, indices = _extract_lag_matrix(
            X, lag_days, self.target, self.lag_indices
        )
        target_values = np.asarray(y, dtype=float)
        if target_values.ndim != 1 or len(target_values) != len(values):
            raise ValueError("y must be one-dimensional and aligned with X.")
        if not np.isfinite(target_values).all():
            raise ValueError("Training labels must be finite; target imputation is forbidden.")
        self.regressor_ = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=True, keep_empty_features=True
                    ),
                ),
                ("scaler", StandardScaler()),
                ("model", LinearRegression(fit_intercept=self.fit_intercept)),
            ]
        )
        self.regressor_.fit(values, target_values)
        self.n_features_in_ = X.shape[1]
        self.lag_columns_ = columns
        self.lag_indices_ = indices
        return self

    def predict(self, X: Any) -> np.ndarray:
        check_is_fitted(self, ("regressor_", "n_features_in_"))
        if isinstance(X, pd.DataFrame) and self.lag_columns_ is not None:
            missing = [column for column in self.lag_columns_ if column not in X.columns]
            if missing:
                raise ValueError(f"Prediction data are missing fitted lag columns: {missing}")
            values = X.loc[:, list(self.lag_columns_)].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=float)
        elif not isinstance(X, pd.DataFrame) and self.lag_indices_ is not None:
            matrix = np.asarray(X, dtype=float)
            if matrix.ndim != 2 or matrix.shape[1] != self.n_features_in_:
                raise ValueError(
                    f"X must have {self.n_features_in_} features; observed shape {matrix.shape}."
                )
            values = matrix[:, list(self.lag_indices_)]
        else:
            raise TypeError("Prediction input type must match the fitted lag representation.")
        return np.asarray(self.regressor_.predict(values), dtype=float).reshape(-1)


def _tabular_pipeline(model: BaseEstimator, *, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def build_tabular_model_pipeline(
    model_name: str,
    *,
    seed: int,
    n_jobs: int = 1,
) -> Pipeline:
    """Build an unfitted table-model pipeline with fold-local preprocessing."""

    name = _canonical_name(model_name)
    if name not in TABULAR_MODEL_NAMES:
        raise ValueError(f"{name} is not a tabular supervised model.")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer.")

    models: dict[str, BaseEstimator] = {
        "ElasticNet": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=20_000,
            tol=1e-5,
            random_state=seed,
            selection="cyclic",
        ),
        "PLS": PLSRegression(n_components=4, scale=False, max_iter=1_000, tol=1e-6),
        "SVR": SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale"),
        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=n_jobs,
        ),
        "XGBoost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        ),
        "LightGBM": LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
        "CatBoost": CatBoostRegressor(
            loss_function="RMSE",
            iterations=400,
            learning_rate=0.05,
            depth=6,
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        ),
    }
    return _tabular_pipeline(models[name], scale=name in SCALED_MODEL_NAMES)


def sample_hyperparameter_candidates(
    model_name: str,
    *,
    candidate_count: int = 12,
    seed: int = 20260815,
    n_features: int | None = None,
) -> tuple[dict[str, Any], ...]:
    """Generate a deterministic, model-specific candidate list.

    Candidate dictionaries can be passed directly to ``set_params``. Persistence
    and seasonal-naive are parameter-free. AutoReg has at most 15 unique subsets
    of the four pre-registered lags, so its returned list is capped at 15.
    """

    _validate_candidate_count(candidate_count)
    name = _canonical_name(model_name)
    rng = _model_rng(seed, name)
    if name in {"Persistence", "SeasonalNaive"}:
        return ({},)
    if name == "AutoReg":
        subsets = [
            tuple(combination)
            for size in range(1, len(DEFAULT_LEGAL_LAGS) + 1)
            for combination in itertools.combinations(DEFAULT_LEGAL_LAGS, size)
        ]
        full = DEFAULT_LEGAL_LAGS
        remainder = [subset for subset in subsets if subset != full]
        rng.shuffle(remainder)
        selected = [full, *remainder[: max(0, candidate_count - 1)]]
        return tuple({"lag_days": subset} for subset in selected)
    if name == "SARIMAX":
        default = {"order": (1, 0, 0), "seasonal_order": (0, 0, 0, 0), "trend": "c"}

        def sarimax_candidate() -> dict[str, Any]:
            seasonal = bool(rng.integers(0, 2))
            return {
                "order": (
                    int(rng.integers(0, 4)),
                    int(rng.integers(0, 2)),
                    int(rng.integers(0, 3)),
                ),
                "seasonal_order": (
                    int(rng.integers(0, 2)),
                    0,
                    int(rng.integers(0, 2)),
                    7,
                )
                if seasonal
                else (0, 0, 0, 0),
                "trend": str(rng.choice(["n", "c"])),
            }

        return _draw_unique_candidates(default, sarimax_candidate, candidate_count)

    feature_cap = max(1, min(int(n_features) if n_features is not None else 10, 10))
    if name == "ElasticNet":
        default = {"model__alpha": 0.01, "model__l1_ratio": 0.5}

        def generator() -> dict[str, Any]:
            return {
                "model__alpha": float(np.round(10 ** rng.uniform(-4.0, 1.0), 9)),
                "model__l1_ratio": float(np.round(rng.uniform(0.05, 0.95), 7)),
            }

    elif name == "PLS":
        default = {"model__n_components": min(4, feature_cap), "model__tol": 1e-6}

        def generator() -> dict[str, Any]:
            return {
                "model__n_components": int(rng.integers(1, feature_cap + 1)),
                "model__tol": float(np.round(10 ** rng.uniform(-9.0, -4.0), 12)),
                "model__max_iter": int(rng.choice([500, 1_000, 2_000, 3_000])),
            }

    elif name == "SVR":
        default = {"model__C": 10.0, "model__epsilon": 0.1, "model__gamma": "scale"}

        def generator() -> dict[str, Any]:
            return {
                "model__C": float(np.round(10 ** rng.uniform(-1.0, 2.5), 8)),
                "model__epsilon": float(np.round(10 ** rng.uniform(-2.5, -0.2), 8)),
                "model__gamma": float(np.round(10 ** rng.uniform(-3.0, 0.0), 9)),
            }

    elif name in {"RandomForest", "ExtraTrees"}:
        default = {
            "model__n_estimators": 400,
            "model__max_depth": None,
            "model__min_samples_leaf": 2,
            "model__max_features": 1.0,
        }

        def generator() -> dict[str, Any]:
            return {
                "model__n_estimators": int(rng.choice([250, 400, 600, 800])),
                "model__max_depth": rng.choice([None, 4, 6, 8, 12]),
                "model__min_samples_leaf": int(rng.choice([1, 2, 4, 8])),
                "model__max_features": float(rng.choice([0.5, 0.7, 0.9, 1.0])),
            }

    elif name == "XGBoost":
        default = {
            "model__n_estimators": 400,
            "model__learning_rate": 0.05,
            "model__max_depth": 4,
            "model__subsample": 0.9,
            "model__colsample_bytree": 0.9,
            "model__reg_lambda": 1.0,
        }

        def generator() -> dict[str, Any]:
            return {
                "model__n_estimators": int(rng.choice([250, 400, 600, 800])),
                "model__learning_rate": float(np.round(10 ** rng.uniform(-2.0, -0.7), 6)),
                "model__max_depth": int(rng.choice([2, 3, 4, 5, 6])),
                "model__subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "model__colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "model__reg_lambda": float(np.round(10 ** rng.uniform(-2.0, 1.5), 7)),
            }

    elif name == "LightGBM":
        default = {
            "model__n_estimators": 400,
            "model__learning_rate": 0.05,
            "model__num_leaves": 31,
            "model__min_child_samples": 20,
            "model__subsample": 0.9,
            "model__colsample_bytree": 0.9,
        }

        def generator() -> dict[str, Any]:
            return {
                "model__n_estimators": int(rng.choice([250, 400, 600, 800])),
                "model__learning_rate": float(np.round(10 ** rng.uniform(-2.0, -0.7), 6)),
                "model__num_leaves": int(rng.choice([15, 31, 63, 127])),
                "model__min_child_samples": int(rng.choice([5, 10, 20, 40])),
                "model__subsample": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
                "model__colsample_bytree": float(rng.choice([0.7, 0.8, 0.9, 1.0])),
            }

    elif name == "CatBoost":
        default = {
            "model__iterations": 400,
            "model__learning_rate": 0.05,
            "model__depth": 6,
            "model__l2_leaf_reg": 3.0,
        }

        def generator() -> dict[str, Any]:
            return {
                "model__iterations": int(rng.choice([250, 400, 600, 800])),
                "model__learning_rate": float(np.round(10 ** rng.uniform(-2.0, -0.7), 6)),
                "model__depth": int(rng.choice([4, 5, 6, 7, 8])),
                "model__l2_leaf_reg": float(np.round(10 ** rng.uniform(-1.0, 1.3), 7)),
            }

    else:  # pragma: no cover - guarded by _canonical_name and branches above
        raise AssertionError(name)
    return _draw_unique_candidates(default, generator, candidate_count)


def build_classical_model_registry(
    *,
    seed: int = 20260815,
    candidate_count: int = 12,
    n_features: int | None = None,
    target: str | None = None,
    n_jobs: int = 1,
) -> dict[str, ModelSpec]:
    """Build the V3 classical/time-series model registry.

    ``SARIMAX`` intentionally has no sklearn estimator: it must be evaluated by
    :func:`rolling_sarimax_one_step` so each forecast precedes state updating.
    """

    _validate_candidate_count(candidate_count)
    families = {
        "Persistence": "naive_baseline",
        "SeasonalNaive": "naive_baseline",
        "AutoReg": "classical_time_series",
        "SARIMAX": "dynamic_regression",
        "ElasticNet": "regularized_linear",
        "PLS": "latent_variable",
        "SVR": "kernel",
        "RandomForest": "bagging",
        "ExtraTrees": "bagging",
        "XGBoost": "boosting",
        "LightGBM": "boosting",
        "CatBoost": "boosting",
    }
    estimators: dict[str, BaseEstimator | None] = {
        "Persistence": PersistenceRegressor(target=target),
        "SeasonalNaive": SeasonalNaiveRegressor(target=target),
        "AutoReg": LegalLagAutoRegressor(target=target),
        "SARIMAX": None,
    }
    estimators.update(
        {
            name: build_tabular_model_pipeline(name, seed=seed, n_jobs=n_jobs)
            for name in TABULAR_MODEL_NAMES
        }
    )
    return {
        name: ModelSpec(
            name=name,
            family=families[name],
            estimator=estimators[name],
            candidates=sample_hyperparameter_candidates(
                name,
                candidate_count=candidate_count,
                seed=seed,
                n_features=n_features,
            ),
            requires_sequential_update=name == "SARIMAX",
        )
        for name in CLASSICAL_MODEL_NAMES
    }


def _as_endog(values: Any, *, name: str, allow_missing: bool) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or not len(array):
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if np.isinf(array).any() or (not allow_missing and np.isnan(array).any()):
        qualifier = "finite or NaN" if allow_missing else "finite"
        raise ValueError(f"{name} values must be {qualifier}.")
    return array


def _as_exog(values: Any, *, name: str, expected_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] != expected_rows or array.shape[1] < 1:
        raise ValueError(
            f"{name} must have shape ({expected_rows}, n_features>=1); observed {array.shape}."
        )
    if np.isinf(array).any():
        raise ValueError(f"{name} cannot contain infinite values.")
    return array


def rolling_sarimax_one_step(
    training_endog: Any,
    evaluation_endog: Any,
    training_exog: Any | None = None,
    evaluation_exog: Any | None = None,
    *,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = "c",
    maxiter: int = 200,
) -> np.ndarray:
    """Return leakage-safe rolling one-step SARIMAX predictions.

    Parameters are fitted once on the training fold. For each evaluation row the
    forecast is emitted first; only then is that row's observed outcome appended
    to the state without refitting. A missing evaluation outcome advances the
    state as missing and is never imputed. Exogenous median imputation and scaling
    are fitted exclusively on ``training_exog``.
    """

    train_y = _as_endog(training_endog, name="training_endog", allow_missing=True)
    evaluation_y = _as_endog(
        evaluation_endog, name="evaluation_endog", allow_missing=True
    )
    minimum_training_observations = max(10, int(order[0]) + int(order[2]) + 2)
    if np.isfinite(train_y).sum() < minimum_training_observations:
        raise ValueError("training_endog has too few observed values for stable SARIMAX fitting.")
    if isinstance(maxiter, bool) or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer.")
    if (training_exog is None) != (evaluation_exog is None):
        raise ValueError("training_exog and evaluation_exog must be supplied together.")

    train_x: np.ndarray | None = None
    evaluation_x: np.ndarray | None = None
    if training_exog is not None:
        raw_train_x = _as_exog(training_exog, name="training_exog", expected_rows=len(train_y))
        raw_evaluation_x = _as_exog(
            evaluation_exog,
            name="evaluation_exog",
            expected_rows=len(evaluation_y),
        )
        if raw_train_x.shape[1] != raw_evaluation_x.shape[1]:
            raise ValueError("Training and evaluation exogenous feature counts differ.")
        exog_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median", add_indicator=False, keep_empty_features=True
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        train_x = exog_pipeline.fit_transform(raw_train_x)
        evaluation_x = exog_pipeline.transform(raw_evaluation_x)

    model = SARIMAX(
        train_y,
        exog=train_x,
        order=tuple(int(value) for value in order),
        seasonal_order=tuple(int(value) for value in seasonal_order),
        trend=trend,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=maxiter)
    predictions = np.empty(len(evaluation_y), dtype=float)
    for index, observed in enumerate(evaluation_y):
        row_x = None if evaluation_x is None else evaluation_x[index : index + 1]
        forecast = fitted.forecast(steps=1, exog=row_x)
        predictions[index] = float(np.asarray(forecast).reshape(-1)[0])
        fitted = fitted.append(
            endog=np.asarray([observed], dtype=float),
            exog=row_x,
            refit=False,
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("SARIMAX produced non-finite predictions.")
    return predictions
