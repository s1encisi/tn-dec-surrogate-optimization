"""Paired post-hoc random interpolation for 2025-only versus three-year training.

The two training windows always face exactly the same shuffled 2025 test dates.
All preprocessing, hyperparameter selection, and stacking are fitted inside the
corresponding outer training subset.  This module intentionally makes no future
forecasting or independent-test claim.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .advanced_prediction import build_advanced_random_registry
from .enhanced_random import (
    EnhancedModelSpec,
    RandomFeatureBundle,
    _fit_base,
    _fit_meta_ensembles,
    make_inner_assignments,
)
from .metrics import regression_metrics
from .model_registry import build_classical_model_registry


TARGETS = ("TN_out", "DEC")
WINDOWS = ("2025_only", "2023_2025")
OUTER_SEEDS = (11, 23, 37, 53, 71)
EXPECTED_SOURCES = {
    "TN_out": ("Q", "COD", "TN", "NH3N", "T", "PPA", "DO", "MLSS"),
    "DEC": ("Q", "COD", "NH3N", "T", "PPA", "DO", "MLSS"),
}
FORBIDDEN_SOURCE_TOKENS = frozenset(
    {"TN_out", "DEC", "TEC", "LTDEC", "PPD", "model_label", "target_ema"}
)
BASE_MODELS = (
    "ElasticNet",
    "PLS",
    "SVR",
    "RandomForest",
    "ExtraTrees",
    "XGBoost",
    "LightGBM",
    "CatBoost",
    "MLP",
    "HistGradientBoosting",
    "GradientBoosting",
    "AdaBoost",
    "KNN",
    "GaussianProcess",
    "TabNet",
)
BASELINES = ("TrainingMean",)
ENSEMBLES = (
    "Ensemble_MeanTopK",
    "Ensemble_Median",
    "Ensemble_NNLS",
    "Ensemble_Ridge",
    "Ensemble_Huber",
)
FEATURE_KEY = "F_INITIAL_ABSOLUTE_TREND"
DATE_ANCHOR = pd.Timestamp("2023-01-01")


class PairedRandomError(ValueError):
    """Raised when the paired-window random protocol would be violated."""


@dataclass(frozen=True)
class PairedFeatureSet:
    bundles: Mapping[str, RandomFeatureBundle]
    common_dates: pd.DatetimeIndex
    feature_registry: pd.DataFrame


@dataclass(frozen=True)
class PairedRandomResult:
    outer_predictions: pd.DataFrame
    inner_oof_predictions: pd.DataFrame
    metrics_by_seed: pd.DataFrame
    leaderboard: pd.DataFrame
    window_comparison_by_seed: pd.DataFrame
    window_comparison_summary: pd.DataFrame
    window_comparison_daily: pd.DataFrame
    tuning_trials: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    ensemble_weights: pd.DataFrame
    outer_assignments: pd.DataFrame
    inner_assignments: pd.DataFrame
    failures: pd.DataFrame
    leakage_audit: pd.DataFrame
    completeness_audit: pd.DataFrame


class TabNetFoldRegressor(RegressorMixin, BaseEstimator):
    """A cloneable TabNet adapter whose early stopping remains training-local."""

    def __init__(
        self,
        *,
        n_d: int = 8,
        n_a: int = 8,
        n_steps: int = 3,
        gamma: float = 1.3,
        lambda_sparse: float = 1e-3,
        max_epochs: int = 80,
        patience: int = 10,
        batch_size: int = 64,
        virtual_batch_size: int = 16,
        random_state: int = 11,
        device_name: str = "cpu",
    ) -> None:
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.random_state = random_state
        self.device_name = device_name

    def fit(self, X: Any, y: Any) -> "TabNetFoldRegressor":
        from pytorch_tabnet.tab_model import TabNetRegressor

        raw_matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=np.float32).reshape(-1)
        if raw_matrix.ndim != 2 or len(raw_matrix) != len(target) or len(target) < 30:
            raise ValueError("TabNet requires at least 30 aligned training rows.")
        if np.isinf(raw_matrix).any() or not np.isfinite(target).all():
            raise ValueError("TabNet adapter received infinite inputs or an invalid target.")
        fit_index, validation_index = train_test_split(
            np.arange(len(target)),
            test_size=0.15,
            shuffle=True,
            random_state=int(self.random_state) + 700_001,
        )
        selection_imputer = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ).fit(raw_matrix[fit_index])
        selection_scaler = StandardScaler().fit(
            selection_imputer.transform(raw_matrix[fit_index])
        )
        fit_matrix = selection_scaler.transform(
            selection_imputer.transform(raw_matrix[fit_index])
        ).astype(np.float32)
        validation_matrix = selection_scaler.transform(
            selection_imputer.transform(raw_matrix[validation_index])
        ).astype(np.float32)
        target_mean = float(target[fit_index].mean())
        target_scale = float(target[fit_index].std(ddof=0))
        if not np.isfinite(target_scale) or target_scale <= 0:
            raise ValueError("TabNet training target has zero or invalid scale.")
        scaled_target = ((target - target_mean) / target_scale).reshape(-1, 1)
        model = TabNetRegressor(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            lambda_sparse=float(self.lambda_sparse),
            seed=int(self.random_state),
            verbose=0,
            device_name=str(self.device_name),
        )
        model.fit(
            fit_matrix,
            scaled_target[fit_index],
            eval_set=[(validation_matrix, scaled_target[validation_index])],
            eval_name=["training_local_validation"],
            eval_metric=["rmse"],
            max_epochs=int(self.max_epochs),
            patience=int(self.patience),
            batch_size=min(int(self.batch_size), len(fit_index)),
            virtual_batch_size=min(int(self.virtual_batch_size), len(fit_index)),
            num_workers=0,
            drop_last=False,
        )
        best_epochs = max(1, int(model.best_epoch) + 1)

        self.imputer_ = SimpleImputer(
            strategy="median", add_indicator=True, keep_empty_features=True
        ).fit(raw_matrix)
        self.scaler_ = StandardScaler().fit(self.imputer_.transform(raw_matrix))
        full_matrix = self.scaler_.transform(self.imputer_.transform(raw_matrix)).astype(
            np.float32
        )
        self.target_mean_ = float(target.mean())
        self.target_scale_ = float(target.std(ddof=0))
        if not np.isfinite(self.target_scale_) or self.target_scale_ <= 0:
            raise ValueError("TabNet full-fold target has zero or invalid scale.")
        full_target = ((target - self.target_mean_) / self.target_scale_).reshape(-1, 1)
        final_model = TabNetRegressor(
            n_d=int(self.n_d),
            n_a=int(self.n_a),
            n_steps=int(self.n_steps),
            gamma=float(self.gamma),
            lambda_sparse=float(self.lambda_sparse),
            seed=int(self.random_state),
            verbose=0,
            device_name=str(self.device_name),
        )
        final_model.fit(
            full_matrix,
            full_target,
            max_epochs=best_epochs,
            patience=0,
            batch_size=min(int(self.batch_size), len(target)),
            virtual_batch_size=min(int(self.virtual_batch_size), len(target)),
            num_workers=0,
            drop_last=False,
        )
        self.model_ = final_model
        self.selected_epochs_ = best_epochs
        self.n_features_in_ = raw_matrix.shape[1]
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "model_"):
            raise ValueError("TabNet adapter is not fitted.")
        raw_matrix = np.asarray(X, dtype=float)
        matrix = self.scaler_.transform(self.imputer_.transform(raw_matrix)).astype(np.float32)
        prediction = np.asarray(self.model_.predict(matrix), dtype=float).reshape(-1)
        return prediction * self.target_scale_ + self.target_mean_


def _normal_mask(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    mapped = series.astype(str).str.strip().str.lower().map({"true": True, "false": False})
    if mapped.isna().any():
        raise PairedRandomError("is_normal_operation contains invalid values.")
    return mapped.astype(bool)


def _validate_daily_data(frame: pd.DataFrame, sources: Mapping[str, Sequence[str]]) -> pd.DataFrame:
    required = {
        "Date",
        "is_normal_operation",
        "study_partition",
        *TARGETS,
        *(column for target in TARGETS for column in sources[target]),
    }
    missing = required.difference(frame.columns)
    if missing:
        raise PairedRandomError(f"Modeling data are missing columns: {sorted(missing)}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.normalize()
    data = data.sort_values("Date").reset_index(drop=True)
    if data["Date"].duplicated().any():
        raise PairedRandomError("Dates must be unique.")
    expected = pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    if not pd.DatetimeIndex(data["Date"]).equals(expected):
        raise PairedRandomError("Input must retain a complete daily calendar.")
    if data["Date"].min() != DATE_ANCHOR or data["Date"].max() < pd.Timestamp("2025-12-31"):
        raise PairedRandomError("Paired comparison requires the complete 2023-2025 calendar.")
    data["is_normal_operation"] = _normal_mask(data["is_normal_operation"])
    for column in required.difference({"Date", "study_partition", "is_normal_operation"}):
        data[column] = pd.to_numeric(data[column], errors="coerce")
        if np.isinf(data[column].to_numpy(float)).any():
            raise PairedRandomError(f"{column} contains infinite values.")
    return data


def build_paired_hrt3_features(
    frame: pd.DataFrame,
    *,
    sources: Mapping[str, Sequence[str]],
    absolute_anchor: str | pd.Timestamp = DATE_ANCHOR,
) -> PairedFeatureSet:
    """Build a common TN/DEC date panel with strict prior-three-day covariates."""

    if tuple(sources) != TARGETS:
        raise PairedRandomError(f"sources must be ordered exactly as {TARGETS}.")
    normalized_sources = {
        target: tuple(str(source) for source in sources[target]) for target in TARGETS
    }
    if normalized_sources != EXPECTED_SOURCES:
        raise PairedRandomError(
            f"HRT3 source registry differs from the exact allowlist: {normalized_sources}"
        )
    anchor_date = pd.Timestamp(absolute_anchor).normalize()
    if anchor_date != DATE_ANCHOR:
        raise PairedRandomError("time_index_days must be anchored to 2023-01-01.")
    data = _validate_daily_data(frame, sources)
    dates = pd.to_datetime(data["Date"])
    normal = data["is_normal_operation"]
    past_normal = normal.astype(int).shift(1).rolling(3, min_periods=3).sum().eq(3)
    common_eligible = normal & past_normal
    for target in TARGETS:
        common_eligible &= data[target].notna()
    common_eligible &= dates.ge(pd.Timestamp("2023-01-04"))
    common_eligible &= ~dates.isin(pd.date_range("2025-01-01", "2025-01-03", freq="D"))

    common_dates = pd.DatetimeIndex(dates.loc[common_eligible])
    if common_dates.duplicated().any() or common_dates.empty:
        raise PairedRandomError("Common paired date panel is empty or duplicated.")
    if common_dates[common_dates.year == 2025].min() != pd.Timestamp("2025-01-04"):
        raise PairedRandomError("2025-only panel borrowed a 2024 warm-up date.")

    bundles: dict[str, RandomFeatureBundle] = {}
    registry_records: list[dict[str, Any]] = []
    day_of_year = dates.dt.dayofyear.astype(float)
    shared = pd.DataFrame(
        {
            "doy_sin": np.sin(2.0 * np.pi * day_of_year / 365.2425),
            "doy_cos": np.cos(2.0 * np.pi * day_of_year / 365.2425),
            "time_index_days": (dates - anchor_date).dt.days.astype(float),
        }
    )
    for target in TARGETS:
        matrix = shared.copy()
        feature_names = ["doy_sin", "doy_cos", "time_index_days"]
        for source in normalized_sources[target]:
            lowered = str(source).lower()
            if (
                source in FORBIDDEN_SOURCE_TOKENS
                or lowered.endswith("_out")
                or "model_label" in lowered
                or "ema" in lowered
            ):
                raise PairedRandomError(f"Forbidden outcome-derived source: {source}")
            feature = f"past3_mean_{source}"
            matrix[feature] = data[source].shift(1).rolling(3, min_periods=3).mean()
            feature_names.append(feature)
        selected_matrix = matrix.loc[common_eligible, feature_names].reset_index(drop=True)
        target_anchor = pd.DataFrame(
            {
                "row_index": np.arange(len(common_dates), dtype=int),
                "Date": common_dates,
                "actual": data.loc[common_eligible, target].to_numpy(float),
                "study_partition": data.loc[common_eligible, "study_partition"].astype(str).to_numpy(),
            }
        )
        bundles[target] = RandomFeatureBundle(
            target=target,
            anchor=target_anchor,
            matrices={FEATURE_KEY: selected_matrix},
            feature_names={FEATURE_KEY: tuple(feature_names)},
            dec_history_reset=None,
        )
        for feature in feature_names:
            if feature in {"doy_sin", "doy_cos"}:
                source = "Date"
                formula = "sin/cos(2*pi*day_of_year/365.2425)"
                lag = 0
            elif feature == "time_index_days":
                source = "Date"
                formula = "Date - 2023-01-01"
                lag = 0
            else:
                source = feature.removeprefix("past3_mean_")
                formula = "mean(x[t-1], x[t-2], x[t-3])"
                lag = 1
            registry_records.append(
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
    tn_dates = pd.DatetimeIndex(bundles["TN_out"].anchor["Date"])
    dec_dates = pd.DatetimeIndex(bundles["DEC"].anchor["Date"])
    if not tn_dates.equals(dec_dates):
        raise RuntimeError("TN_out and DEC do not share an identical date anchor.")
    return PairedFeatureSet(
        bundles=bundles,
        common_dates=common_dates,
        feature_registry=pd.DataFrame.from_records(registry_records),
    )


def make_shared_2025_assignments(
    common_dates: Sequence[pd.Timestamp],
    *,
    seeds: Sequence[int] = OUTER_SEEDS,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Generate one ordinary row-random 2025 assignment table for both targets/windows."""

    normalized_seeds = tuple(int(seed) for seed in seeds)
    if normalized_seeds != OUTER_SEEDS:
        raise PairedRandomError(f"Outer seeds must be exactly {OUTER_SEEDS}.")
    if not np.isclose(float(test_fraction), 0.20):
        raise PairedRandomError("The outer split must be ordinary random 80:20.")
    dates = pd.DatetimeIndex(pd.to_datetime(common_dates))
    pool = dates[(dates >= pd.Timestamp("2025-01-04")) & (dates <= pd.Timestamp("2025-12-31"))]
    if pool.duplicated().any() or not len(pool):
        raise PairedRandomError("The common 2025 test pool is empty or duplicated.")
    records: list[pd.DataFrame] = []
    positions = np.arange(len(pool), dtype=int)
    for seed in normalized_seeds:
        train, test = train_test_split(
            positions,
            test_size=float(test_fraction),
            random_state=seed,
            shuffle=True,
        )
        role = np.full(len(pool), "outer_train", dtype=object)
        role[test] = "outer_test"
        if np.intersect1d(train, test).size:
            raise RuntimeError("Outer train/test overlap.")
        records.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "Date": pool,
                    "role": role,
                    "ordinary_daily_random": True,
                }
            )
        )
    return pd.concat(records, ignore_index=True)


def _tabnet_pipeline(tabnet_config: Mapping[str, Any], seed: int) -> Pipeline:
    return Pipeline(
        [
            (
                "model",
                TabNetFoldRegressor(
                    max_epochs=int(tabnet_config["max_epochs"]),
                    patience=int(tabnet_config["patience"]),
                    batch_size=int(tabnet_config["batch_size"]),
                    virtual_batch_size=int(tabnet_config["virtual_batch_size"]),
                    random_state=seed,
                    device_name=str(tabnet_config["device_name"]),
                ),
            ),
        ]
    )


def build_paired_model_registry(
    *,
    seed: int,
    n_features: int,
    n_jobs: int,
    candidate_count: int,
    tabnet_config: Mapping[str, Any],
) -> dict[str, EnhancedModelSpec]:
    """Build the exact 15-model pool plus a pre-fixed mean baseline."""

    if candidate_count < 1:
        raise PairedRandomError("candidate_count must be positive.")
    classical = build_classical_model_registry(
        seed=seed,
        candidate_count=candidate_count,
        n_features=n_features,
        target=None,
        n_jobs=n_jobs,
    )
    advanced = build_advanced_random_registry(seed)
    specs: dict[str, EnhancedModelSpec] = {}
    classical_names = (
        "ElasticNet",
        "PLS",
        "SVR",
        "RandomForest",
        "ExtraTrees",
        "XGBoost",
        "LightGBM",
        "CatBoost",
    )
    for name in classical_names:
        source = classical[name]
        if source.estimator is None:
            raise RuntimeError(f"{name} has no registered estimator.")
        specs[name] = EnhancedModelSpec(
            name,
            "registered_classic_or_boosting",
            FEATURE_KEY,
            FEATURE_KEY,
            "MATERIALIZED_INPUT",
            source.estimator,
            source.candidates,
        )

    mlp = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
            ),
            ("scaler", StandardScaler()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=MLPRegressor(
                        random_state=seed,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=25,
                        max_iter=1_000,
                    ),
                    transformer=StandardScaler(),
                ),
            ),
        ]
    )
    mlp_candidates = (
        {"model__regressor__hidden_layer_sizes": (32,), "model__regressor__alpha": 1e-4},
        {"model__regressor__hidden_layer_sizes": (64, 32), "model__regressor__alpha": 1e-3},
        {"model__regressor__hidden_layer_sizes": (64, 32, 16), "model__regressor__alpha": 1e-2},
        {"model__regressor__hidden_layer_sizes": (128, 64), "model__regressor__alpha": 1e-3},
    )[:candidate_count]
    specs["MLP"] = EnhancedModelSpec(
        "MLP",
        "multilayer_perceptron",
        FEATURE_KEY,
        FEATURE_KEY,
        "MATERIALIZED_INPUT",
        mlp,
        mlp_candidates,
    )
    for name in (
        "HistGradientBoosting",
        "GradientBoosting",
        "AdaBoost",
        "KNN",
        "GaussianProcess",
    ):
        source = advanced[name]
        specs[name] = EnhancedModelSpec(
            name,
            "advanced_registered_regressor",
            FEATURE_KEY,
            FEATURE_KEY,
            "MATERIALIZED_INPUT",
            source.estimator,
            source.candidates[:candidate_count],
        )
    raw_tabnet_candidates = tabnet_config.get("candidates")
    if not isinstance(raw_tabnet_candidates, list) or not raw_tabnet_candidates:
        raise PairedRandomError("TabNet candidates must be an explicit non-empty list.")
    tabnet_candidates = tuple(
        {f"model__{key}": value for key, value in candidate.items()}
        for candidate in raw_tabnet_candidates[:candidate_count]
    )
    specs["TabNet"] = EnhancedModelSpec(
        "TabNet",
        "tabnet",
        FEATURE_KEY,
        FEATURE_KEY,
        "MATERIALIZED_INPUT",
        _tabnet_pipeline(tabnet_config, seed),
        tabnet_candidates,
    )
    if tuple(specs) != BASE_MODELS:
        raise RuntimeError(f"Base registry differs from the frozen 15-model order: {tuple(specs)}")
    specs["TrainingMean"] = EnhancedModelSpec(
        "TrainingMean",
        "training_mean_baseline",
        FEATURE_KEY,
        "NONE_TARGET_MEAN",
        "NONE",
        DummyRegressor(strategy="mean"),
        ({},),
        ensemble_eligible=False,
        analysis_role="pre_registered_baseline",
    )
    return specs


def _indices_for_window(
    bundle: RandomFeatureBundle,
    assignment: pd.DataFrame,
    window: str,
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(bundle.anchor["Date"]))
    test_dates = set(pd.to_datetime(assignment.loc[assignment["role"].eq("outer_test"), "Date"]))
    train_2025_dates = set(
        pd.to_datetime(assignment.loc[assignment["role"].eq("outer_train"), "Date"])
    )
    test = np.flatnonzero(dates.isin(test_dates))
    if window == "2025_only":
        train = np.flatnonzero(dates.isin(train_2025_dates))
    elif window == "2023_2025":
        train = np.flatnonzero((dates < pd.Timestamp("2025-01-01")) | dates.isin(train_2025_dates))
    else:
        raise PairedRandomError(f"Unknown training window: {window}")
    if np.intersect1d(train, test).size:
        raise RuntimeError("Window training data include an outer-test row.")
    return train.astype(int), test.astype(int)


def _prediction_rows(
    bundle: RandomFeatureBundle,
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    window: str,
    seed: int,
    model: str,
    model_role: str,
    family: str,
) -> pd.DataFrame:
    selected = bundle.anchor.iloc[np.asarray(indices, dtype=int)]
    values = np.asarray(prediction, dtype=float).reshape(-1)
    if len(selected) != len(values) or not np.isfinite(values).all():
        raise RuntimeError(f"Invalid predictions for {bundle.target}/{window}/{model}/{seed}.")
    return pd.DataFrame(
        {
            "target": bundle.target,
            "training_window": window,
            "seed": seed,
            "model": model,
            "model_role": model_role,
            "model_family": family,
            "Date": pd.to_datetime(selected["Date"]).to_numpy(),
            "actual": selected["actual"].to_numpy(float),
            "prediction": values,
            "analysis_role": "post_hoc_same_distribution_random_interpolation",
            "future_prediction_claim_authorized": False,
            "independent_test": False,
        }
    )


def _metric_tables(predictions: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_seed_records: list[dict[str, Any]] = []
    grouping = ["target", "training_window", "model", "model_role", "model_family", "seed"]
    for keys, group in predictions.groupby(grouping, observed=True, sort=False):
        metrics = regression_metrics(group["actual"].to_numpy(float), group["prediction"].to_numpy(float))
        by_seed_records.append(dict(zip(grouping, keys, strict=True)) | metrics)
    by_seed = pd.DataFrame.from_records(by_seed_records)
    metrics_to_summarize = (
        "R2",
        "RMSE",
        "MAE",
        "MAPE_pct",
        "sMAPE_pct",
        "WAPE_pct",
        "PBIAS_pct",
        "Q90_abs_error",
    )
    summary_records: list[dict[str, Any]] = []
    summary_grouping = ["target", "training_window", "model", "model_role", "model_family"]
    for keys, group in by_seed.groupby(summary_grouping, observed=True, sort=False):
        record = dict(zip(summary_grouping, keys, strict=True))
        record["n_seeds"] = int(len(group))
        for metric in metrics_to_summarize:
            values = pd.to_numeric(group[metric], errors="coerce")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
            record[f"{metric}_min"] = float(values.min())
            record[f"{metric}_max"] = float(values.max())
        summary_records.append(record)
    summary = pd.DataFrame.from_records(summary_records)
    summary = summary.sort_values(
        ["target", "training_window", "model_role", "RMSE_mean", "R2_mean"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)
    summary["rank_within_role"] = (
        summary.groupby(["target", "training_window", "model_role"], observed=True).cumcount() + 1
    )
    return by_seed, summary


def _paired_window_comparison(
    metrics_by_seed: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ("R2", "RMSE", "MAE", "MAPE_pct", "WAPE_pct")
    identifier = ["target", "model", "model_role", "model_family", "seed"]
    wide = metrics_by_seed.pivot(index=identifier, columns="training_window", values=list(metrics))
    for window in WINDOWS:
        if window not in wide.columns.get_level_values(1):
            raise RuntimeError(f"Missing paired metrics for {window}.")
    records: list[dict[str, Any]] = []
    for index, row in wide.iterrows():
        record = dict(zip(identifier, index, strict=True))
        for metric in metrics:
            y2025 = float(row[(metric, "2025_only")])
            three_year = float(row[(metric, "2023_2025")])
            record[f"{metric}_2025_only"] = y2025
            record[f"{metric}_2023_2025"] = three_year
            record[f"delta_{metric}_three_year_minus_2025"] = three_year - y2025
        records.append(record)
    by_seed = pd.DataFrame.from_records(records)
    rng = np.random.default_rng(int(bootstrap_seed))
    summary_records: list[dict[str, Any]] = []
    grouping = ["target", "model", "model_role", "model_family"]
    for keys, group in by_seed.groupby(grouping, observed=True, sort=False):
        record = dict(zip(grouping, keys, strict=True))
        record["n_seeds"] = int(len(group))
        for metric in metrics:
            column = f"delta_{metric}_three_year_minus_2025"
            delta = group[column].to_numpy(float)
            draws = rng.choice(delta, size=(int(bootstrap_replicates), len(delta)), replace=True).mean(axis=1)
            record[f"{column}_mean"] = float(delta.mean())
            record[f"{column}_sd"] = float(delta.std(ddof=1))
            record[f"{column}_descriptive_seed_resampling95_low"] = float(
                np.quantile(draws, 0.025)
            )
            record[f"{column}_descriptive_seed_resampling95_high"] = float(
                np.quantile(draws, 0.975)
            )
        record["three_year_RMSE_wins"] = int(
            (group["delta_RMSE_three_year_minus_2025"] < 0).sum()
        )
        record["interval_role"] = (
            "descriptive_seed_resampling_only; five overlapping random splits are not independent"
        )
        summary_records.append(record)
    return by_seed, pd.DataFrame.from_records(summary_records)


def _daily_window_comparison(predictions: pd.DataFrame) -> pd.DataFrame:
    identifiers = ["target", "model", "model_role", "model_family", "seed", "Date"]
    left = predictions.loc[
        predictions["training_window"].eq("2025_only"),
        [*identifiers, "actual", "prediction"],
    ]
    right = predictions.loc[
        predictions["training_window"].eq("2023_2025"),
        [*identifiers, "actual", "prediction"],
    ]
    paired = left.merge(
        right,
        on=identifiers,
        how="inner",
        validate="one_to_one",
        suffixes=("_2025_only", "_2023_2025"),
    )
    if len(paired) != len(left) or len(paired) != len(right):
        raise RuntimeError("Training windows do not share exactly the same prediction dates.")
    np.testing.assert_allclose(
        paired["actual_2025_only"].to_numpy(float),
        paired["actual_2023_2025"].to_numpy(float),
        rtol=0.0,
        atol=0.0,
    )
    paired["actual"] = paired.pop("actual_2025_only")
    paired = paired.drop(columns=["actual_2023_2025"])
    residual_2025 = paired["actual"] - paired["prediction_2025_only"]
    residual_three_year = paired["actual"] - paired["prediction_2023_2025"]
    paired["delta_squared_error_three_year_minus_2025"] = (
        residual_three_year**2 - residual_2025**2
    )
    paired["delta_absolute_error_three_year_minus_2025"] = (
        residual_three_year.abs() - residual_2025.abs()
    )
    return paired.sort_values(identifiers).reset_index(drop=True)


def run_paired_random_windows(
    feature_set: PairedFeatureSet,
    registries: Mapping[str, Mapping[str, EnhancedModelSpec]],
    assignments: pd.DataFrame,
    *,
    inner_folds: int,
    ensemble_model_ids: Sequence[str],
    top_k_candidates: Sequence[int],
    ridge_alphas: Sequence[float],
    bootstrap_replicates: int,
    bootstrap_seed: int,
    progress: Callable[[str], None] | None = None,
) -> PairedRandomResult:
    """Run all target/window/seed combinations against shared 2025 test dates."""

    requested_ensemble = tuple(str(value) for value in ensemble_model_ids)
    if requested_ensemble != BASE_MODELS:
        raise PairedRandomError("All and only the frozen 15 base models must enter stacking.")
    prediction_frames: list[pd.DataFrame] = []
    inner_prediction_frames: list[pd.DataFrame] = []
    tuning_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    inner_assignment_frames: list[pd.DataFrame] = []
    failure_records: list[dict[str, Any]] = []
    leakage_records: list[dict[str, Any]] = []

    for target in TARGETS:
        bundle = feature_set.bundles[target]
        registry = registries[target]
        if tuple(registry) != (*BASE_MODELS, *BASELINES):
            raise PairedRandomError(f"Incomplete model registry for {target}: {tuple(registry)}")
        y = bundle.anchor["actual"].to_numpy(float)
        for window in WINDOWS:
            for seed in OUTER_SEEDS:
                if progress is not None:
                    progress(f"target={target} window={window} seed={seed}: preparing folds")
                assignment = assignments.loc[assignments["seed"].eq(seed)].copy()
                outer_train, outer_test = _indices_for_window(bundle, assignment, window)
                folds, inner_table = make_inner_assignments(
                    bundle, outer_train, seed=seed, n_splits=inner_folds
                )
                inner_table.insert(1, "training_window", window)
                inner_assignment_frames.append(inner_table)
                fitted_models: dict[str, Any] = {}
                for model, spec in registry.items():
                    if progress is not None:
                        progress(f"target={target} window={window} seed={seed}: fitting {model}")
                    fitted = _fit_base(
                        bundle,
                        spec,
                        outer_train,
                        outer_test,
                        folds,
                        seed=seed,
                    )
                    fitted_models[model] = fitted
                    tuning_records.extend(
                        {"training_window": window, **record} for record in fitted.tuning_records
                    )
                    role = "base_model" if model in BASE_MODELS else "pre_registered_baseline"
                    selected_records.append(
                        {
                            "target": target,
                            "training_window": window,
                            "seed": seed,
                            "model": model,
                            "model_role": role,
                            "selected_candidate": fitted.selected_candidate,
                            "selected_parameters": json.dumps(
                                fitted.selected_parameters, ensure_ascii=False, default=str
                            ),
                            "selection_source": "outer_train_three_fold_oof_only",
                            "outer_test_accessed_for_selection": False,
                        }
                    )
                    inner_prediction_frames.append(
                        _prediction_rows(
                            bundle,
                            outer_train,
                            fitted.inner_oof_prediction,
                            window=window,
                            seed=seed,
                            model=model,
                            model_role=role,
                            family=spec.family,
                        )
                    )
                    prediction_frames.append(
                        _prediction_rows(
                            bundle,
                            outer_test,
                            fitted.outer_test_prediction,
                            window=window,
                            seed=seed,
                            model=model,
                            model_role=role,
                            family=spec.family,
                        )
                    )

                base_oof = pd.DataFrame(
                    {model: fitted_models[model].inner_oof_prediction for model in requested_ensemble}
                )
                base_test = pd.DataFrame(
                    {model: fitted_models[model].outer_test_prediction for model in requested_ensemble}
                )
                if not base_oof.notna().all().all():
                    raise RuntimeError("A base model lacks strict inner OOF predictions.")
                meta_predictions, meta_weights, meta_failures = _fit_meta_ensembles(
                    y[outer_train],
                    base_oof,
                    base_test,
                    top_k_candidates=top_k_candidates,
                    ridge_alphas=ridge_alphas,
                )
                if progress is not None:
                    progress(f"target={target} window={window} seed={seed}: ensembles complete")
                if tuple(meta_predictions) != ENSEMBLES:
                    raise RuntimeError(f"Incomplete ensemble output: {tuple(meta_predictions)}")
                for record in meta_weights:
                    weight_records.append(
                        {"target": target, "training_window": window, "seed": seed, **record}
                    )
                for record in meta_failures:
                    failure_records.append(
                        {"target": target, "training_window": window, "seed": seed, **record}
                    )
                for model, prediction in meta_predictions.items():
                    prediction_frames.append(
                        _prediction_rows(
                            bundle,
                            outer_test,
                            prediction,
                            window=window,
                            seed=seed,
                            model=model,
                            model_role="oof_ensemble",
                            family="strict_outer_train_oof_ensemble",
                        )
                    )
                dates = pd.DatetimeIndex(pd.to_datetime(bundle.anchor["Date"]))
                test_hash = hashlib.sha256(
                    "\n".join(dates[outer_test].strftime("%Y-%m-%d")).encode()
                ).hexdigest()
                leakage_records.append(
                    {
                        "target": target,
                        "training_window": window,
                        "seed": seed,
                        "outer_train_n": len(outer_train),
                        "outer_test_n": len(outer_test),
                        "outer_overlap_n": int(np.intersect1d(outer_train, outer_test).size),
                        "pre_2025_train_n": int((dates[outer_train] < pd.Timestamp("2025-01-01")).sum()),
                        "test_date_sha256": test_hash,
                        "inner_oof_complete_all_15": bool(base_oof.notna().all().all()),
                        "target_history_columns_used": 0,
                        "effluent_feature_columns_used": 0,
                        "outer_test_accessed_for_tuning_or_weights": False,
                        "preprocessing_fit_scope": "inner_or_outer_training_only",
                        "ordinary_daily_random": True,
                        "future_prediction_claim_authorized": False,
                        "independent_test": False,
                    }
                )

    outer_predictions = pd.concat(prediction_frames, ignore_index=True)
    inner_predictions = pd.concat(inner_prediction_frames, ignore_index=True)
    metrics_by_seed, leaderboard = _metric_tables(outer_predictions)
    comparison_by_seed, comparison_summary = _paired_window_comparison(
        metrics_by_seed,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    comparison_daily = _daily_window_comparison(outer_predictions)
    expected_models = len(BASE_MODELS) + len(BASELINES) + len(ENSEMBLES)
    group_columns = ["target", "training_window", "seed", "model"]
    groups = outer_predictions.groupby(
        group_columns, observed=True
    ).size()
    expected_groups = len(TARGETS) * len(WINDOWS) * len(OUTER_SEEDS) * expected_models
    expected_model_set = set((*BASE_MODELS, *BASELINES, *ENSEMBLES))
    exact_model_sets = outer_predictions.groupby(
        ["target", "training_window", "seed"], observed=True
    )["model"].agg(lambda values: set(values) == expected_model_set)
    unique_dates = outer_predictions.groupby(group_columns, observed=True)["Date"].agg(
        lambda values: not pd.Series(values).duplicated().any()
    )
    leakage = pd.DataFrame.from_records(leakage_records)
    shared_test_hash = leakage.groupby("seed", observed=True)["test_date_sha256"].nunique().eq(1)
    all_groups_complete = bool(
        len(groups) == expected_groups
        and groups.eq(70).all()
        and unique_dates.all()
        and exact_model_sets.all()
        and shared_test_hash.all()
    )
    completeness = pd.DataFrame(
        [
            {
                "expected_model_groups": expected_groups,
                "observed_model_groups": int(len(groups)),
                "all_groups_complete": all_groups_complete,
                "expected_models_per_target_window_seed": expected_models,
                "minimum_test_rows_per_group": int(groups.min()),
                "maximum_test_rows_per_group": int(groups.max()),
                "all_groups_have_exactly_70_rows": bool(groups.eq(70).all()),
                "all_group_dates_unique": bool(unique_dates.all()),
                "all_model_sets_exact": bool(exact_model_sets.all()),
                "all_targets_windows_share_test_hash_per_seed": bool(shared_test_hash.all()),
                "daily_window_pairs_complete": bool(len(comparison_daily) * 2 == len(outer_predictions)),
                "base_model_count": len(BASE_MODELS),
                "ensemble_count": len(ENSEMBLES),
                "baseline_count": len(BASELINES),
            }
        ]
    )
    if not bool(completeness.loc[0, "all_groups_complete"]):
        raise RuntimeError("The paired comparison is incomplete; no leaderboard may be published.")
    return PairedRandomResult(
        outer_predictions=outer_predictions,
        inner_oof_predictions=inner_predictions,
        metrics_by_seed=metrics_by_seed,
        leaderboard=leaderboard,
        window_comparison_by_seed=comparison_by_seed,
        window_comparison_summary=comparison_summary,
        window_comparison_daily=comparison_daily,
        tuning_trials=pd.DataFrame.from_records(tuning_records),
        selected_hyperparameters=pd.DataFrame.from_records(selected_records),
        ensemble_weights=pd.DataFrame.from_records(weight_records),
        outer_assignments=assignments.copy(),
        inner_assignments=pd.concat(inner_assignment_frames, ignore_index=True),
        failures=pd.DataFrame.from_records(failure_records),
        leakage_audit=leakage,
        completeness_audit=completeness,
    )
