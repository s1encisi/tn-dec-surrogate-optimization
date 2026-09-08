"""Leakage-safe enhanced random-interpolation experiments.

The outer 80:20 test subset is used only once, after all base-model tuning and
ensemble fitting have completed inside the outer training subset.  Main models
use the registered F-A covariates only: target history, rolling target summaries,
and other effluent measurements are intentionally absent.

Two date-only diagnostics are included to quantify how much a shuffled split can
benefit from observations on both sides of a test date.  They are explicitly
excluded from every ensemble and from any future-prediction recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from .features import build_tabular_next_day_features, target_feature_spec
from .metrics import regression_metrics


OUTER_SEEDS = (11, 23, 37, 53, 71)
MAIN_ANALYSIS_ROLE = "enhanced_random_interpolation"
DATE_DIAGNOSTIC_ROLE = "bidirectional_random_interpolation_only"
ENSEMBLE_FEATURE_VERSION = "STACKED_REGISTERED_NO_TARGET_HISTORY"
ALLOWED_ALIGNMENTS = ("L0", "L1", "L2")
REGISTERED_FEATURE_KEYS = (
    "F_HRT3_TREND",
    "F_A_L0",
    "F_A_L1",
    "F_A_L2",
)


class EnhancedRandomError(ValueError):
    """Raised when the enhanced random protocol would violate its contract."""


@dataclass(frozen=True)
class RandomFeatureBundle:
    """Paired F-A matrices for the same dates under registered alignments."""

    target: str
    anchor: pd.DataFrame
    matrices: Mapping[str, pd.DataFrame]
    feature_names: Mapping[str, tuple[str, ...]]
    dec_history_reset: str | None

    def matrix(self, feature_key: str) -> pd.DataFrame:
        if feature_key == "DATE_ONLY":
            dates = pd.to_datetime(self.anchor["Date"])
            return pd.DataFrame({"date_ordinal": dates.map(pd.Timestamp.toordinal).astype(float)})
        try:
            return self.matrices[feature_key]
        except KeyError as exc:
            raise EnhancedRandomError(f"Unavailable feature key: {feature_key}") from exc


@dataclass(frozen=True)
class EnhancedModelSpec:
    model_id: str
    family: str
    feature_key: str
    feature_version: str
    alignment: str
    estimator: RegressorMixin
    candidates: tuple[dict[str, Any], ...]
    ensemble_eligible: bool = True
    analysis_role: str = MAIN_ANALYSIS_ROLE


@dataclass(frozen=True)
class FittedBase:
    model_id: str
    inner_oof_prediction: np.ndarray
    outer_test_prediction: np.ndarray
    selected_candidate: int
    selected_parameters: dict[str, Any]
    tuning_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class EnhancedRandomResult:
    outer_predictions: pd.DataFrame
    inner_oof_predictions: pd.DataFrame
    metrics_by_seed: pd.DataFrame
    leaderboard: pd.DataFrame
    tuning_trials: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    ensemble_weights: pd.DataFrame
    outer_assignments: pd.DataFrame
    inner_assignments: pd.DataFrame
    failures: pd.DataFrame
    leakage_audit: pd.DataFrame


class BidirectionalLinearDateInterpolator(RegressorMixin, BaseEstimator):
    """Linear interpolation on date, with nearest-value boundary extrapolation."""

    def fit(self, X: Any, y: Any) -> BidirectionalLinearDateInterpolator:
        dates = _date_vector(X)
        target = np.asarray(y, dtype=float).reshape(-1)
        if len(dates) != len(target) or len(dates) < 2:
            raise ValueError("Date interpolation requires at least two aligned observations.")
        if not np.isfinite(dates).all() or not np.isfinite(target).all():
            raise ValueError("Date interpolation requires finite dates and targets.")
        order = np.argsort(dates)
        self.dates_ = dates[order]
        self.target_ = target[order]
        if np.unique(self.dates_).size != len(self.dates_):
            raise ValueError("Date interpolation requires unique dates.")
        self.n_features_in_ = 1
        return self

    def predict(self, X: Any) -> np.ndarray:
        if not hasattr(self, "dates_"):
            raise ValueError("Date interpolator is not fitted.")
        dates = _date_vector(X)
        return np.interp(dates, self.dates_, self.target_).astype(float)


def _date_vector(X: Any) -> np.ndarray:
    if isinstance(X, pd.DataFrame):
        if list(X.columns) != ["date_ordinal"]:
            raise ValueError("Date-only models require the date_ordinal column only.")
        values = X["date_ordinal"].to_numpy(float)
    else:
        matrix = np.asarray(X, dtype=float)
        values = matrix.reshape(-1) if matrix.ndim <= 1 else matrix[:, 0]
    return np.asarray(values, dtype=float).reshape(-1)


def _validate_daily_development(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    required = {"Date", "study_partition", "is_normal_operation", target}
    missing = required.difference(frame.columns)
    if missing:
        raise EnhancedRandomError(f"Missing required columns: {sorted(missing)}")
    data = frame.copy()
    data["Date"] = pd.to_datetime(data["Date"], errors="raise").dt.normalize()
    data = data.sort_values("Date").reset_index(drop=True)
    if data["Date"].duplicated().any():
        raise EnhancedRandomError("Dates must be unique.")
    if len(data) and not pd.DatetimeIndex(data["Date"]).equals(
        pd.date_range(data["Date"].min(), data["Date"].max(), freq="D")
    ):
        raise EnhancedRandomError("Input must retain the complete daily calendar.")
    if set(data["study_partition"].astype(str)) != {"development"}:
        raise EnhancedRandomError("Enhanced random experiments are development-only.")
    data[target] = pd.to_numeric(data[target], errors="coerce")
    if np.isinf(data[target].to_numpy(float)).any():
        raise EnhancedRandomError("Target contains infinite values.")
    return data


def _build_hrt3_trend_panel(
    data: pd.DataFrame,
    feature_config: dict[str, Any],
    target: str,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Build known calendar/trend plus strict t-1:t-3 exogenous means."""

    spec = target_feature_spec(feature_config, target)
    dates = pd.to_datetime(data["Date"])
    actual = pd.to_numeric(data[target], errors="coerce")
    panel = pd.DataFrame(
        {
            "Date": dates,
            "actual": actual,
            "study_partition": data["study_partition"],
        }
    )
    day_of_year = dates.dt.dayofyear.astype(float)
    panel["doy_sin"] = np.sin(2.0 * np.pi * day_of_year / 365.2425)
    panel["doy_cos"] = np.cos(2.0 * np.pi * day_of_year / 365.2425)
    panel["time_index_days"] = (dates - dates.min()).dt.days.astype(float)
    feature_names = ["doy_sin", "doy_cos", "time_index_days"]
    exogenous = [*spec["g2_influent"], *spec["g3_process"]]
    for column in exogenous:
        if str(column).lower().endswith("_out") or column == target:
            raise EnhancedRandomError(
                f"HRT3 registry contains a forbidden effluent variable: {column}"
            )
        name = f"past3_mean_{column}"
        values = pd.to_numeric(data[column], errors="coerce")
        panel[name] = values.shift(1).rolling(window=3, min_periods=3).mean()
        feature_names.append(name)
    normal = data["is_normal_operation"].fillna(False).astype(bool)
    past_normal = (
        normal.astype(int).shift(1).rolling(window=3, min_periods=3).sum().eq(3)
    )
    panel["eligible"] = normal & past_normal & actual.notna()
    maximum = int(
        feature_config["constraints"]["maximum_table_features_before_missing_indicators"]
    )
    if len(feature_names) > maximum:
        raise EnhancedRandomError(
            f"HRT3 feature count {len(feature_names)} exceeds limit {maximum}."
        )
    return panel, tuple(feature_names)


def build_random_feature_bundle(
    development_data: pd.DataFrame,
    feature_config: dict[str, Any],
    target: str,
    *,
    feature_keys: Sequence[str] = REGISTERED_FEATURE_KEYS,
    dec_history_reset: str = "2025-01-01",
) -> RandomFeatureBundle:
    """Build paired target-history-free matrices for the registered feature keys."""

    data = _validate_daily_development(development_data, target)
    normalized_keys = tuple(str(value) for value in feature_keys)
    if normalized_keys != REGISTERED_FEATURE_KEYS:
        raise EnhancedRandomError(
            f"Feature keys must be exactly {REGISTERED_FEATURE_KEYS}."
        )
    reset_value: str | None = None
    if target == "DEC":
        reset = pd.Timestamp(dec_history_reset).normalize()
        if data["Date"].min() != reset:
            raise EnhancedRandomError(
                "DEC input must begin exactly at the registered 2025-01-01 reset."
            )
        reset_value = reset.date().isoformat()

    panels: dict[str, pd.DataFrame] = {}
    names_by_key: dict[str, tuple[str, ...]] = {}
    eligible_dates: list[set[pd.Timestamp]] = []
    for feature_key in normalized_keys:
        if feature_key == "F_HRT3_TREND":
            panel, raw_names = _build_hrt3_trend_panel(data, feature_config, target)
        else:
            alignment = feature_key.removeprefix("F_A_")
            panel, raw_names = build_tabular_next_day_features(
                data, feature_config, target, "F_A", alignment
            )
        names = tuple(str(name) for name in raw_names)
        forbidden = [
            name
            for name in names
            if name.startswith("lag")
            or "rolling" in name.lower()
            or name == target
            or name.lower().endswith("_out")
        ]
        if forbidden:
            raise EnhancedRandomError(
                f"Random features contain forbidden outcome history: {forbidden}"
            )
        eligible = panel["eligible"].astype(bool) & panel["actual"].notna()
        selected = panel.loc[
            eligible, ["Date", "actual", "study_partition", *names]
        ].copy()
        if selected.empty:
            raise EnhancedRandomError(f"No eligible rows for {target}/{feature_key}.")
        selected["Date"] = pd.to_datetime(selected["Date"])
        panels[feature_key] = selected
        names_by_key[feature_key] = names
        eligible_dates.append(set(selected["Date"]))

    common_dates = sorted(set.intersection(*eligible_dates))
    if len(common_dates) < 50:
        raise EnhancedRandomError("Fewer than 50 dates are paired across feature keys.")
    first = normalized_keys[0]
    reference = panels[first].set_index("Date").loc[common_dates]
    anchor = reference[["actual", "study_partition"]].reset_index()
    matrices: dict[str, pd.DataFrame] = {}
    for feature_key in normalized_keys:
        indexed = panels[feature_key].set_index("Date").loc[common_dates]
        np.testing.assert_allclose(
            indexed["actual"].to_numpy(float), anchor["actual"].to_numpy(float)
        )
        matrices[feature_key] = indexed.loc[:, list(names_by_key[feature_key])].reset_index(
            drop=True
        )
    anchor.insert(0, "row_index", np.arange(len(anchor), dtype=int))
    return RandomFeatureBundle(
        target=target,
        anchor=anchor,
        matrices=matrices,
        feature_names=names_by_key,
        dec_history_reset=reset_value,
    )


def _tabular_pipeline(model: RegressorMixin, *, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median", add_indicator=True, keep_empty_features=True
            ),
        )
    ]
    if scale:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)


def _tree_candidates() -> tuple[dict[str, Any], ...]:
    return (
        {
            "model__n_estimators": 250,
            "model__max_depth": None,
            "model__min_samples_leaf": 1,
            "model__max_features": 0.65,
        },
        {
            "model__n_estimators": 350,
            "model__max_depth": 8,
            "model__min_samples_leaf": 1,
            "model__max_features": 1.0,
        },
        {
            "model__n_estimators": 450,
            "model__max_depth": 12,
            "model__min_samples_leaf": 2,
            "model__max_features": 0.8,
        },
        {
            "model__n_estimators": 600,
            "model__max_depth": None,
            "model__min_samples_leaf": 4,
            "model__max_features": 1.0,
        },
    )


def build_enhanced_registry(
    seed: int,
    feature_keys_by_model: Mapping[str, str],
    *,
    n_jobs: int = 1,
) -> dict[str, EnhancedModelSpec]:
    """Build the preregistered diverse base pool and excluded date diagnostics."""

    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0:
        raise EnhancedRandomError("n_jobs must be a non-zero integer.")
    expected = (
        "ExtraTrees_HRT3",
        "RandomForest_HRT3",
        "XGBoost_HRT3",
        "XGB_DART_HRT3",
        "LightGBM_HRT3",
        "LGBM_DART_HRT3",
        "CatBoost_HRT3",
        "HistGradientBoosting_HRT3",
        "GBHuber_HRT3",
        "SVR_HRT3",
        "ExtraTrees_FA_L0",
        "ExtraTrees_FA_L1",
        "ExtraTrees_FA_L2",
    )
    if tuple(feature_keys_by_model) != expected:
        raise EnhancedRandomError(
            "Enhanced model/feature registry differs from the preregistered order."
        )
    invalid = {model: key for model, key in feature_keys_by_model.items() if key not in REGISTERED_FEATURE_KEYS}
    if invalid:
        raise EnhancedRandomError(f"Invalid registered feature keys: {invalid}")
    for model in expected[:10]:
        if feature_keys_by_model[model] != "F_HRT3_TREND":
            raise EnhancedRandomError("All primary model families must use F_HRT3_TREND.")
    expected_ablation = {
        "ExtraTrees_FA_L0": "F_A_L0",
        "ExtraTrees_FA_L1": "F_A_L1",
        "ExtraTrees_FA_L2": "F_A_L2",
    }
    if any(feature_keys_by_model[model] != key for model, key in expected_ablation.items()):
        raise EnhancedRandomError("ExtraTrees F-A alignment ablation is not preregistered.")

    extra = _tabular_pipeline(
        ExtraTreesRegressor(random_state=seed, n_jobs=n_jobs), scale=False
    )
    forest = _tabular_pipeline(
        RandomForestRegressor(random_state=seed, n_jobs=n_jobs), scale=False
    )
    xgb = _tabular_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            booster="gbtree",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
        ),
        scale=False,
    )
    xgb_dart = _tabular_pipeline(
        XGBRegressor(
            objective="reg:squarederror",
            booster="dart",
            random_state=seed,
            n_jobs=n_jobs,
            tree_method="hist",
            verbosity=0,
            sample_type="uniform",
            normalize_type="tree",
        ),
        scale=False,
    )
    lightgbm = _tabular_pipeline(
        LGBMRegressor(
            objective="regression",
            boosting_type="gbdt",
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
        scale=False,
    )
    lightgbm_dart = _tabular_pipeline(
        LGBMRegressor(
            objective="regression",
            boosting_type="dart",
            random_state=seed,
            n_jobs=n_jobs,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        ),
        scale=False,
    )
    catboost = _tabular_pipeline(
        CatBoostRegressor(
            loss_function="RMSE",
            random_seed=seed,
            thread_count=n_jobs,
            verbose=False,
            allow_writing_files=False,
        ),
        scale=False,
    )
    hist = _tabular_pipeline(
        HistGradientBoostingRegressor(random_state=seed, early_stopping=True),
        scale=False,
    )
    gb_huber = _tabular_pipeline(
        GradientBoostingRegressor(random_state=seed, loss="huber"), scale=False
    )
    svr = _tabular_pipeline(SVR(kernel="rbf"), scale=True)

    common_xgb = (
        {
            "model__n_estimators": 180,
            "model__learning_rate": 0.06,
            "model__max_depth": 2,
            "model__min_child_weight": 5,
            "model__subsample": 0.8,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 5.0,
        },
        {
            "model__n_estimators": 260,
            "model__learning_rate": 0.04,
            "model__max_depth": 3,
            "model__min_child_weight": 3,
            "model__subsample": 0.85,
            "model__colsample_bytree": 0.85,
            "model__reg_lambda": 2.0,
        },
        {
            "model__n_estimators": 380,
            "model__learning_rate": 0.025,
            "model__max_depth": 4,
            "model__min_child_weight": 5,
            "model__subsample": 0.9,
            "model__colsample_bytree": 1.0,
            "model__reg_lambda": 5.0,
        },
        {
            "model__n_estimators": 500,
            "model__learning_rate": 0.02,
            "model__max_depth": 2,
            "model__min_child_weight": 8,
            "model__subsample": 1.0,
            "model__colsample_bytree": 0.75,
            "model__reg_lambda": 10.0,
        },
    )
    dart_candidates = tuple(
        {
            **candidate,
            "model__rate_drop": rate,
            "model__skip_drop": skip,
        }
        for candidate, rate, skip in zip(
            common_xgb,
            (0.02, 0.05, 0.10, 0.15),
            (0.0, 0.05, 0.0, 0.10),
            strict=True,
        )
    )
    common_lgbm = (
        {
            "model__n_estimators": 200,
            "model__learning_rate": 0.05,
            "model__num_leaves": 7,
            "model__min_child_samples": 20,
            "model__subsample": 0.85,
            "model__colsample_bytree": 0.85,
            "model__reg_lambda": 5.0,
        },
        {
            "model__n_estimators": 300,
            "model__learning_rate": 0.035,
            "model__num_leaves": 15,
            "model__min_child_samples": 10,
            "model__subsample": 0.9,
            "model__colsample_bytree": 0.9,
            "model__reg_lambda": 2.0,
        },
        {
            "model__n_estimators": 420,
            "model__learning_rate": 0.025,
            "model__num_leaves": 31,
            "model__min_child_samples": 20,
            "model__subsample": 1.0,
            "model__colsample_bytree": 0.8,
            "model__reg_lambda": 5.0,
        },
        {
            "model__n_estimators": 520,
            "model__learning_rate": 0.02,
            "model__num_leaves": 15,
            "model__min_child_samples": 30,
            "model__subsample": 0.8,
            "model__colsample_bytree": 1.0,
            "model__reg_lambda": 10.0,
        },
    )
    dart_lgbm = tuple(
        {**candidate, "model__drop_rate": rate, "model__skip_drop": 0.0}
        for candidate, rate in zip(
            common_lgbm, (0.02, 0.05, 0.10, 0.15), strict=True
        )
    )
    specifications = (
        ("ExtraTrees_HRT3", "bagging_extra_trees", extra, _tree_candidates()),
        (
            "RandomForest_HRT3",
            "bagging_random_forest",
            forest,
            _tree_candidates(),
        ),
        ("XGBoost_HRT3", "boosting_xgboost_gbtree", xgb, common_xgb),
        ("XGB_DART_HRT3", "boosting_xgboost_dart", xgb_dart, dart_candidates),
        ("LightGBM_HRT3", "boosting_lightgbm_gbdt", lightgbm, common_lgbm),
        ("LGBM_DART_HRT3", "boosting_lightgbm_dart", lightgbm_dart, dart_lgbm),
        (
            "CatBoost_HRT3",
            "boosting_catboost",
            catboost,
            (
                {
                    "model__iterations": 250,
                    "model__learning_rate": 0.05,
                    "model__depth": 4,
                    "model__l2_leaf_reg": 3.0,
                    "model__boosting_type": "Plain",
                },
                {
                    "model__iterations": 300,
                    "model__learning_rate": 0.04,
                    "model__depth": 5,
                    "model__l2_leaf_reg": 5.0,
                    "model__boosting_type": "Ordered",
                },
                {
                    "model__iterations": 450,
                    "model__learning_rate": 0.025,
                    "model__depth": 6,
                    "model__l2_leaf_reg": 7.0,
                    "model__boosting_type": "Plain",
                },
                {
                    "model__iterations": 500,
                    "model__learning_rate": 0.02,
                    "model__depth": 6,
                    "model__l2_leaf_reg": 10.0,
                    "model__boosting_type": "Ordered",
                },
            ),
        ),
        (
            "HistGradientBoosting_HRT3",
            "boosting_hist_gradient",
            hist,
            (
                {
                    "model__max_iter": 180,
                    "model__learning_rate": 0.06,
                    "model__max_leaf_nodes": 7,
                    "model__min_samples_leaf": 10,
                    "model__l2_regularization": 0.5,
                },
                {
                    "model__max_iter": 250,
                    "model__learning_rate": 0.04,
                    "model__max_leaf_nodes": 15,
                    "model__min_samples_leaf": 10,
                    "model__l2_regularization": 1.0,
                },
                {
                    "model__max_iter": 320,
                    "model__learning_rate": 0.03,
                    "model__max_leaf_nodes": 31,
                    "model__min_samples_leaf": 20,
                    "model__l2_regularization": 3.0,
                },
                {
                    "model__max_iter": 400,
                    "model__learning_rate": 0.02,
                    "model__max_leaf_nodes": 15,
                    "model__min_samples_leaf": 30,
                    "model__l2_regularization": 8.0,
                },
            ),
        ),
        (
            "GBHuber_HRT3",
            "boosting_gradient_huber",
            gb_huber,
            (
                {
                    "model__n_estimators": 180,
                    "model__learning_rate": 0.05,
                    "model__max_depth": 1,
                    "model__min_samples_leaf": 5,
                },
                {
                    "model__n_estimators": 250,
                    "model__learning_rate": 0.035,
                    "model__max_depth": 2,
                    "model__min_samples_leaf": 5,
                },
                {
                    "model__n_estimators": 350,
                    "model__learning_rate": 0.025,
                    "model__max_depth": 2,
                    "model__min_samples_leaf": 10,
                },
                {
                    "model__n_estimators": 450,
                    "model__learning_rate": 0.02,
                    "model__max_depth": 3,
                    "model__min_samples_leaf": 15,
                },
            ),
        ),
        (
            "SVR_HRT3",
            "kernel_rbf",
            svr,
            (
                {"model__C": 1.0, "model__epsilon": 0.05, "model__gamma": "scale"},
                {"model__C": 5.0, "model__epsilon": 0.10, "model__gamma": 0.03},
                {"model__C": 15.0, "model__epsilon": 0.05, "model__gamma": 0.10},
                {"model__C": 40.0, "model__epsilon": 0.15, "model__gamma": 0.30},
            ),
        ),
        ("ExtraTrees_FA_L0", "bagging_extra_trees_ablation", extra, _tree_candidates()),
        ("ExtraTrees_FA_L1", "bagging_extra_trees_ablation", extra, _tree_candidates()),
        ("ExtraTrees_FA_L2", "bagging_extra_trees_control", extra, _tree_candidates()),
    )
    registry: dict[str, EnhancedModelSpec] = {}
    for model_id, family, estimator, candidates in specifications:
        feature_key = feature_keys_by_model[model_id]
        feature_version = "F_A" if feature_key.startswith("F_A_") else feature_key
        alignment = (
            feature_key.removeprefix("F_A_")
            if feature_key.startswith("F_A_")
            else "HRT3_T1_T3_MEAN"
        )
        registry[model_id] = EnhancedModelSpec(
            model_id,
            family,
            feature_key,
            feature_version,
            alignment,
            estimator,
            candidates,
        )

    date_knn = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", KNeighborsRegressor(weights="distance", p=1)),
        ]
    )
    registry["TimeKNN"] = EnhancedModelSpec(
        "TimeKNN",
        "bidirectional_time_knn",
        "DATE_ONLY",
        "DATE_ONLY",
        "DATE_ONLY",
        date_knn,
        tuple({"model__n_neighbors": value} for value in (2, 4, 8, 14)),
        ensemble_eligible=False,
        analysis_role=DATE_DIAGNOSTIC_ROLE,
    )
    registry["LinearDateInterpolation"] = EnhancedModelSpec(
        "LinearDateInterpolation",
        "bidirectional_linear_date_interpolation",
        "DATE_ONLY",
        "DATE_ONLY",
        "DATE_ONLY",
        BidirectionalLinearDateInterpolator(),
        ({},),
        ensemble_eligible=False,
        analysis_role=DATE_DIAGNOSTIC_ROLE,
    )
    registry["TrainingMean"] = EnhancedModelSpec(
        "TrainingMean",
        "training_mean_baseline",
        "F_HRT3_TREND",
        "NONE_TARGET_MEAN",
        "NONE",
        DummyRegressor(strategy="mean"),
        ({},),
        ensemble_eligible=False,
        analysis_role=MAIN_ANALYSIS_ROLE,
    )
    return registry


def make_outer_assignments(
    bundle: RandomFeatureBundle,
    *,
    seeds: Sequence[int] = OUTER_SEEDS,
    test_fraction: float = 0.20,
) -> pd.DataFrame:
    """Create paired 80:20 assignments for the five preregistered seeds."""

    normalized = tuple(int(seed) for seed in seeds)
    if normalized != OUTER_SEEDS:
        raise EnhancedRandomError(f"Outer seeds must be exactly {OUTER_SEEDS}.")
    if not np.isclose(test_fraction, 0.20):
        raise EnhancedRandomError("Outer random split must be exactly 80:20.")
    indices = np.arange(len(bundle.anchor), dtype=int)
    records: list[pd.DataFrame] = []
    for seed in normalized:
        train, test = train_test_split(
            indices, test_size=test_fraction, random_state=seed, shuffle=True
        )
        role = np.full(len(indices), "outer_train", dtype=object)
        role[test] = "outer_test"
        records.append(
            pd.DataFrame(
                {
                    "target": bundle.target,
                    "seed": seed,
                    "row_index": indices,
                    "Date": pd.to_datetime(bundle.anchor["Date"]),
                    "role": role,
                }
            )
        )
        if np.intersect1d(train, test).size:  # pragma: no cover - sklearn invariant
            raise RuntimeError("Outer train/test overlap.")
    return pd.concat(records, ignore_index=True)


def make_inner_assignments(
    bundle: RandomFeatureBundle,
    outer_train: np.ndarray,
    *,
    seed: int,
    n_splits: int = 3,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """Return three shuffled folds defined exclusively inside outer training."""

    train_indices = np.asarray(outer_train, dtype=int)
    if n_splits != 3:
        raise EnhancedRandomError("Enhanced ensembles require exactly three inner folds.")
    if len(train_indices) < 3 * n_splits:
        raise EnhancedRandomError("Outer training subset is too small for three inner folds.")
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed + 300_003)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[pd.DataFrame] = []
    for fold_index, (fit_position, validation_position) in enumerate(
        splitter.split(train_indices), start=1
    ):
        fit_indices = train_indices[fit_position]
        validation_indices = train_indices[validation_position]
        folds.append((fit_indices, validation_indices))
        role = np.full(len(train_indices), "inner_train", dtype=object)
        role[validation_position] = "inner_validation"
        records.append(
            pd.DataFrame(
                {
                    "target": bundle.target,
                    "seed": seed,
                    "inner_fold": fold_index,
                    "row_index": train_indices,
                    "Date": pd.to_datetime(
                        bundle.anchor.iloc[train_indices]["Date"]
                    ).to_numpy(),
                    "role": role,
                }
            )
        )
    validation = np.concatenate([fold[1] for fold in folds])
    if set(validation) != set(train_indices) or len(validation) != len(train_indices):
        raise RuntimeError("Inner OOF validation rows are not an exact outer-train partition.")
    return folds, pd.concat(records, ignore_index=True)


def _fit_base(
    bundle: RandomFeatureBundle,
    spec: EnhancedModelSpec,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    seed: int,
) -> FittedBase:
    X = bundle.matrix(spec.feature_key)
    y = bundle.anchor["actual"].to_numpy(float)
    outer_train = np.asarray(outer_train, dtype=int)
    outer_test = np.asarray(outer_test, dtype=int)
    train_position = {int(index): position for position, index in enumerate(outer_train)}
    candidate_predictions: list[np.ndarray] = []
    tuning_records: list[dict[str, Any]] = []

    def seeded_clone() -> RegressorMixin:
        estimator = clone(spec.estimator)
        available = estimator.get_params(deep=True)
        seed_parameters = {
            key: seed
            for key in available
            if key.endswith("random_state") or key.endswith("random_seed")
        }
        return estimator.set_params(**seed_parameters)

    for candidate_index, parameters in enumerate(spec.candidates):
        oof = np.full(len(outer_train), np.nan, dtype=float)
        candidate_failed = False
        for fold_index, (fit_indices, validation_indices) in enumerate(
            inner_folds, start=1
        ):
            try:
                fitted = seeded_clone().set_params(**parameters)
                fitted.fit(X.iloc[fit_indices], y[fit_indices])
                prediction = np.asarray(
                    fitted.predict(X.iloc[validation_indices]), dtype=float
                ).reshape(-1)
                if len(prediction) != len(validation_indices) or not np.isfinite(
                    prediction
                ).all():
                    raise RuntimeError("Candidate returned invalid validation predictions.")
                positions = [train_position[int(index)] for index in validation_indices]
                oof[positions] = prediction
                rmse = float(
                    mean_squared_error(y[validation_indices], prediction) ** 0.5
                )
                mae = float(mean_absolute_error(y[validation_indices], prediction))
                status = "completed"
                error = None
            except Exception as exc:
                candidate_failed = True
                rmse = np.nan
                mae = np.nan
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            tuning_records.append(
                {
                    "target": bundle.target,
                    "seed": seed,
                    "model": spec.model_id,
                    "model_family": spec.family,
                    "feature_version": spec.feature_version,
                    "alignment": spec.alignment,
                    "candidate_index": candidate_index,
                    "parameters": json.dumps(parameters, ensure_ascii=False, default=str),
                    "record_type": "inner_fold",
                    "inner_fold": fold_index,
                    "inner_RMSE": rmse,
                    "inner_MAE": mae,
                    "status": status,
                    "error": error,
                    "outer_test_accessed": False,
                }
            )
        if candidate_failed or not np.isfinite(oof).all():
            candidate_predictions.append(np.full_like(oof, np.nan))
            continue
        pooled_rmse = float(mean_squared_error(y[outer_train], oof) ** 0.5)
        pooled_mae = float(mean_absolute_error(y[outer_train], oof))
        tuning_records.append(
            {
                "target": bundle.target,
                "seed": seed,
                "model": spec.model_id,
                "model_family": spec.family,
                "feature_version": spec.feature_version,
                "alignment": spec.alignment,
                "candidate_index": candidate_index,
                "parameters": json.dumps(parameters, ensure_ascii=False, default=str),
                "record_type": "pooled_inner_oof",
                "inner_fold": np.nan,
                "inner_RMSE": pooled_rmse,
                "inner_MAE": pooled_mae,
                "status": "completed",
                "error": None,
                "outer_test_accessed": False,
            }
        )
        candidate_predictions.append(oof)

    completed = [
        record
        for record in tuning_records
        if record["record_type"] == "pooled_inner_oof"
        and record["status"] == "completed"
    ]
    if not completed:
        raise RuntimeError(f"All candidates failed for {spec.model_id}/seed={seed}.")
    best_rmse = min(float(record["inner_RMSE"]) for record in completed)
    near = [
        record
        for record in completed
        if float(record["inner_RMSE"]) <= best_rmse * 1.01
    ]
    selected = min(
        near,
        key=lambda record: (
            float(record["inner_MAE"]),
            int(record["candidate_index"]),
        ),
    )
    selected_index = int(selected["candidate_index"])
    selected_parameters = dict(spec.candidates[selected_index])
    final = seeded_clone().set_params(**selected_parameters)
    final.fit(X.iloc[outer_train], y[outer_train])
    outer_prediction = np.asarray(final.predict(X.iloc[outer_test]), dtype=float).reshape(-1)
    if len(outer_prediction) != len(outer_test) or not np.isfinite(
        outer_prediction
    ).all():
        raise RuntimeError(f"{spec.model_id} returned invalid outer-test predictions.")
    return FittedBase(
        model_id=spec.model_id,
        inner_oof_prediction=candidate_predictions[selected_index],
        outer_test_prediction=outer_prediction,
        selected_candidate=selected_index,
        selected_parameters=selected_parameters,
        tuning_records=tuple(tuning_records),
    )


def _metric_pair(y: np.ndarray, prediction: np.ndarray) -> tuple[float, float]:
    return (
        float(mean_squared_error(y, prediction) ** 0.5),
        float(mean_absolute_error(y, prediction)),
    )


def _fit_meta_ensembles(
    y_train: np.ndarray,
    base_oof: pd.DataFrame,
    base_test: pd.DataFrame,
    *,
    top_k_candidates: Sequence[int],
    ridge_alphas: Sequence[float],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fit ensemble rules on inner OOF columns only and predict the outer test."""

    P = base_oof.to_numpy(float)
    P_test = base_test.loc[:, base_oof.columns].to_numpy(float)
    y = np.asarray(y_train, dtype=float)
    if P.shape[0] != len(y) or P.shape[1] < 2:
        raise EnhancedRandomError("At least two paired base OOF columns are required.")
    if not np.isfinite(P).all() or not np.isfinite(P_test).all():
        raise EnhancedRandomError("Meta learning requires finite paired base predictions.")
    model_ids = list(base_oof.columns)
    predictions: dict[str, np.ndarray] = {}
    weights: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    base_rmse = {
        model: float(mean_squared_error(y, base_oof[model]) ** 0.5)
        for model in model_ids
    }
    ordered = sorted(model_ids, key=lambda model: (base_rmse[model], model))
    valid_k = sorted(
        {
            min(max(1, int(value)), len(model_ids))
            for value in top_k_candidates
        }
    )
    top_trials: list[tuple[float, float, int, list[str]]] = []
    for k in valid_k:
        selected = ordered[:k]
        train_prediction = base_oof[selected].mean(axis=1).to_numpy(float)
        rmse, mae = _metric_pair(y, train_prediction)
        top_trials.append((rmse, mae, k, selected))
    best_rmse = min(trial[0] for trial in top_trials)
    top_near = [trial for trial in top_trials if trial[0] <= best_rmse * 1.01]
    _, _, selected_k, selected_models = min(
        top_near, key=lambda trial: (trial[1], trial[2])
    )
    predictions["Ensemble_MeanTopK"] = (
        base_test[selected_models].mean(axis=1).to_numpy(float)
    )
    for model in model_ids:
        weights.append(
            {
                "ensemble_model": "Ensemble_MeanTopK",
                "component_model": model,
                "weight": 1.0 / selected_k if model in selected_models else 0.0,
                "intercept": 0.0,
                "meta_parameter": f"selected_k={selected_k}",
            }
        )

    predictions["Ensemble_Median"] = np.median(P_test, axis=1)
    for model in model_ids:
        weights.append(
            {
                "ensemble_model": "Ensemble_Median",
                "component_model": model,
                "weight": np.nan,
                "intercept": np.nan,
                "meta_parameter": "median_aggregation",
            }
        )

    centered_P = P - P.mean(axis=0, keepdims=True)
    centered_y = y - y.mean()
    nnls_weights, _ = nnls(centered_P, centered_y)
    nnls_intercept = float(y.mean() - P.mean(axis=0) @ nnls_weights)
    predictions["Ensemble_NNLS"] = nnls_intercept + P_test @ nnls_weights
    for model, weight in zip(model_ids, nnls_weights, strict=True):
        weights.append(
            {
                "ensemble_model": "Ensemble_NNLS",
                "component_model": model,
                "weight": float(weight),
                "intercept": nnls_intercept,
                "meta_parameter": "nonnegative_least_squares_centered",
            }
        )

    alphas = np.asarray(tuple(float(value) for value in ridge_alphas), dtype=float)
    if not len(alphas) or (alphas <= 0).any():
        raise EnhancedRandomError("Ridge alphas must be positive.")
    ridge = RidgeCV(alphas=alphas, fit_intercept=True).fit(P, y)
    predictions["Ensemble_Ridge"] = np.asarray(ridge.predict(P_test), dtype=float)
    for model, weight in zip(model_ids, ridge.coef_, strict=True):
        weights.append(
            {
                "ensemble_model": "Ensemble_Ridge",
                "component_model": model,
                "weight": float(weight),
                "intercept": float(ridge.intercept_),
                "meta_parameter": f"alpha={float(ridge.alpha_):.12g}",
            }
        )

    try:
        scaler = StandardScaler().fit(P)
        scaled_P = scaler.transform(P)
        huber = HuberRegressor(
            epsilon=1.35, alpha=0.001, max_iter=2_000, tol=1e-7
        ).fit(scaled_P, y)
        predictions["Ensemble_Huber"] = np.asarray(
            huber.predict(scaler.transform(P_test)), dtype=float
        )
        effective = huber.coef_ / scaler.scale_
        intercept = float(huber.intercept_ - np.dot(effective, scaler.mean_))
        for model, weight in zip(model_ids, effective, strict=True):
            weights.append(
                {
                    "ensemble_model": "Ensemble_Huber",
                    "component_model": model,
                    "weight": float(weight),
                    "intercept": intercept,
                    "meta_parameter": "epsilon=1.35;alpha=0.001",
                }
            )
    except Exception as exc:  # numerical collinearity is audited, not hidden
        failures.append(
            {
                "model": "Ensemble_Huber",
                "stage": "meta_fit",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    for model, prediction in predictions.items():
        if not np.isfinite(prediction).all():
            raise RuntimeError(f"{model} returned non-finite outer-test predictions.")
    return predictions, weights, failures


def _prediction_rows(
    bundle: RandomFeatureBundle,
    indices: np.ndarray,
    prediction: np.ndarray,
    *,
    seed: int,
    model: str,
    family: str,
    feature_version: str,
    alignment: str,
    analysis_role: str,
    ensemble_eligible: bool,
) -> pd.DataFrame:
    selected = bundle.anchor.iloc[np.asarray(indices, dtype=int)]
    return pd.DataFrame(
        {
            "target": bundle.target,
            "model": model,
            "model_family": family,
            "feature_version": feature_version,
            "alignment": alignment,
            "seed": seed,
            "row_index": selected["row_index"].to_numpy(int),
            "Date": pd.to_datetime(selected["Date"]).to_numpy(),
            "actual": selected["actual"].to_numpy(float),
            "prediction": np.asarray(prediction, dtype=float),
            "analysis_role": analysis_role,
            "ensemble_eligible": bool(ensemble_eligible),
            "future_prediction_claim_authorized": False,
        }
    )


def _metrics_and_summary(
    predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    group_columns = [
        "target",
        "model",
        "model_family",
        "feature_version",
        "alignment",
        "analysis_role",
        "ensemble_eligible",
        "seed",
    ]
    metric_records: list[dict[str, Any]] = []
    for keys, group in predictions.groupby(group_columns, observed=True, sort=False):
        record = dict(zip(group_columns, keys, strict=True))
        record.update(
            regression_metrics(
                group["actual"].to_numpy(float), group["prediction"].to_numpy(float)
            )
        )
        metric_records.append(record)
    metrics = pd.DataFrame.from_records(metric_records)
    summary_columns = group_columns[:-1]
    summary_records: list[dict[str, Any]] = []
    metric_names = ("R2", "RMSE", "MAE", "MAPE_pct", "WAPE_pct")
    for keys, group in metrics.groupby(summary_columns, observed=True, sort=False):
        record = dict(zip(summary_columns, keys, strict=True))
        record["n_seeds"] = int(len(group))
        for metric in metric_names:
            values = pd.to_numeric(group[metric], errors="raise")
            record[f"{metric}_mean"] = float(values.mean())
            record[f"{metric}_sd"] = float(values.std(ddof=1))
        record["n_seed_R2_ge_0_85"] = int(group["R2"].ge(0.85).sum())
        summary_records.append(record)
    summary = pd.DataFrame.from_records(summary_records)
    summary = summary.sort_values(
        ["analysis_role", "R2_mean", "RMSE_mean"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    summary["rank_within_analysis_role"] = (
        summary.groupby("analysis_role", observed=True).cumcount() + 1
    )
    return metrics, summary


def run_enhanced_random(
    bundle: RandomFeatureBundle,
    registry: Mapping[str, EnhancedModelSpec],
    *,
    outer_seeds: Sequence[int] = OUTER_SEEDS,
    test_fraction: float = 0.20,
    inner_folds: int = 3,
    ensemble_model_ids: Sequence[str],
    top_k_candidates: Sequence[int] = (3, 5, 8, 12),
    ridge_alphas: Sequence[float] = (
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
    ),
) -> EnhancedRandomResult:
    """Run the complete nested-OOF random interpolation protocol."""

    assignments = make_outer_assignments(
        bundle, seeds=outer_seeds, test_fraction=test_fraction
    )
    requested_ensemble = tuple(str(value) for value in ensemble_model_ids)
    if not requested_ensemble or len(set(requested_ensemble)) != len(
        requested_ensemble
    ):
        raise EnhancedRandomError("ensemble_model_ids must be non-empty and unique.")
    unknown = set(requested_ensemble).difference(registry)
    if unknown:
        raise EnhancedRandomError(f"Unknown ensemble models: {sorted(unknown)}")
    ineligible = [
        model for model in requested_ensemble if not registry[model].ensemble_eligible
    ]
    if ineligible:
        raise EnhancedRandomError(
            f"Date-only diagnostics cannot enter an ensemble: {ineligible}"
        )

    outer_prediction_frames: list[pd.DataFrame] = []
    inner_prediction_frames: list[pd.DataFrame] = []
    tuning_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    inner_assignment_frames: list[pd.DataFrame] = []
    failures: list[dict[str, Any]] = []
    leakage_records: list[dict[str, Any]] = []
    y = bundle.anchor["actual"].to_numpy(float)

    for seed in tuple(int(value) for value in outer_seeds):
        split = assignments.loc[assignments["seed"].eq(seed)]
        outer_train = split.loc[split["role"].eq("outer_train"), "row_index"].to_numpy(int)
        outer_test = split.loc[split["role"].eq("outer_test"), "row_index"].to_numpy(int)
        folds, inner_table = make_inner_assignments(
            bundle, outer_train, seed=seed, n_splits=inner_folds
        )
        inner_assignment_frames.append(inner_table)
        fitted_bases: dict[str, FittedBase] = {}
        for model_id, spec in registry.items():
            fitted = _fit_base(
                bundle,
                spec,
                outer_train,
                outer_test,
                folds,
                seed=seed,
            )
            fitted_bases[model_id] = fitted
            tuning_records.extend(fitted.tuning_records)
            selected_records.append(
                {
                    "target": bundle.target,
                    "seed": seed,
                    "model": model_id,
                    "model_family": spec.family,
                    "feature_version": spec.feature_version,
                    "alignment": spec.alignment,
                    "analysis_role": spec.analysis_role,
                    "ensemble_eligible": spec.ensemble_eligible,
                    "selected_candidate": fitted.selected_candidate,
                    "selected_parameters": json.dumps(
                        fitted.selected_parameters,
                        ensure_ascii=False,
                        default=str,
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
                    seed=seed,
                    model=model_id,
                    family=spec.family,
                    feature_version=spec.feature_version,
                    alignment=spec.alignment,
                    analysis_role=spec.analysis_role,
                    ensemble_eligible=spec.ensemble_eligible,
                )
            )
            outer_prediction_frames.append(
                _prediction_rows(
                    bundle,
                    outer_test,
                    fitted.outer_test_prediction,
                    seed=seed,
                    model=model_id,
                    family=spec.family,
                    feature_version=spec.feature_version,
                    alignment=spec.alignment,
                    analysis_role=spec.analysis_role,
                    ensemble_eligible=spec.ensemble_eligible,
                )
            )

        base_oof = pd.DataFrame(
            {
                model: fitted_bases[model].inner_oof_prediction
                for model in requested_ensemble
            }
        )
        base_test = pd.DataFrame(
            {
                model: fitted_bases[model].outer_test_prediction
                for model in requested_ensemble
            }
        )
        meta_predictions, meta_weights, meta_failures = _fit_meta_ensembles(
            y[outer_train],
            base_oof,
            base_test,
            top_k_candidates=top_k_candidates,
            ridge_alphas=ridge_alphas,
        )
        for record in meta_weights:
            weight_records.append({"target": bundle.target, "seed": seed, **record})
        for record in meta_failures:
            failures.append({"target": bundle.target, "seed": seed, **record})
        for model, prediction in meta_predictions.items():
            outer_prediction_frames.append(
                _prediction_rows(
                    bundle,
                    outer_test,
                    prediction,
                    seed=seed,
                    model=model,
                    family="inner_oof_stacking_or_aggregation",
                    feature_version=ENSEMBLE_FEATURE_VERSION,
                    alignment="MULTI_REGISTERED",
                    analysis_role=MAIN_ANALYSIS_ROLE,
                    ensemble_eligible=False,
                )
            )
        leakage_records.append(
            {
                "target": bundle.target,
                "seed": seed,
                "outer_train_n": len(outer_train),
                "outer_test_n": len(outer_test),
                "outer_overlap_n": int(np.intersect1d(outer_train, outer_test).size),
                "inner_oof_rows": len(base_oof),
                "inner_oof_complete": bool(base_oof.notna().all().all()),
                "main_feature_version": "F_HRT3_TREND",
                "comparison_feature_version": "F_A_L2",
                "outcome_derived_feature_count": 0,
                "same_day_effluent_feature_count": 0,
                "lagged_effluent_feature_count": 0,
                "target_history_columns_used": 0,
                "rolling_target_columns_used": 0,
                "other_effluent_columns_used": 0,
                "date_diagnostics_in_ensemble": False,
                "ensemble_fit_scope": "outer_train_inner_oof_only",
                "outer_test_accessed_for_tuning_or_weights": False,
                "future_prediction_claim_authorized": False,
                "dec_history_reset": bundle.dec_history_reset,
            }
        )

    outer_predictions = pd.concat(outer_prediction_frames, ignore_index=True)
    inner_predictions = pd.concat(inner_prediction_frames, ignore_index=True)
    metrics, leaderboard = _metrics_and_summary(outer_predictions)
    return EnhancedRandomResult(
        outer_predictions=outer_predictions,
        inner_oof_predictions=inner_predictions,
        metrics_by_seed=metrics,
        leaderboard=leaderboard,
        tuning_trials=pd.DataFrame.from_records(tuning_records),
        selected_hyperparameters=pd.DataFrame.from_records(selected_records),
        ensemble_weights=pd.DataFrame.from_records(weight_records),
        outer_assignments=assignments,
        inner_assignments=pd.concat(inner_assignment_frames, ignore_index=True),
        failures=pd.DataFrame.from_records(failures),
        leakage_audit=pd.DataFrame.from_records(leakage_records),
    )
