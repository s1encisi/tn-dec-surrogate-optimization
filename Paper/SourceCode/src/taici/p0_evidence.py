"""P0 evidence-boundary audits for temporal prediction and proxy-RL uncertainty.

The analyses in this module are deliberately post hoc.  They strengthen the
diagnostic evidence boundary but do not create an independent test set, a
calibrated counterfactual interval, or plant-control evidence.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from .enhanced_random import EnhancedModelSpec, RandomFeatureBundle
from .metrics import regression_metrics
from .paired_random_windows import BASE_MODELS, PairedFeatureSet


FINAL_MODELS = {"TN_out": "Ensemble_Huber", "DEC": "ExtraTrees"}
RL_METHODS = ("PPO", "SAC", "TD3")


class P0EvidenceError(ValueError):
    """Raised when a P0 audit contract would be violated."""


@dataclass(frozen=True)
class OuterFold:
    name: str
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class TemporalBaseFit:
    model: str
    selected_candidate: int
    selected_parameters: Mapping[str, Any]
    validation_indices: np.ndarray
    validation_prediction: np.ndarray
    test_prediction: np.ndarray
    tuning_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class TemporalValidationResult:
    predictions: pd.DataFrame
    metrics_by_fold: pd.DataFrame
    pooled_metrics: pd.DataFrame
    selected_hyperparameters: pd.DataFrame
    tuning_trials: pd.DataFrame
    ensemble_weights: pd.DataFrame
    outer_assignments: pd.DataFrame
    inner_assignments: pd.DataFrame
    leakage_audit: pd.DataFrame


@dataclass(frozen=True)
class ProxyUncertaintyResult:
    residual_library: pd.DataFrame
    residual_scale: pd.DataFrame
    episode_intervals: pd.DataFrame
    method_summary: pd.DataFrame


def parse_outer_folds(records: Sequence[Mapping[str, Any]]) -> tuple[OuterFold, ...]:
    folds = tuple(
        OuterFold(
            name=str(record["name"]),
            train_end=pd.Timestamp(record["train_end"]).normalize(),
            test_start=pd.Timestamp(record["test_start"]).normalize(),
            test_end=pd.Timestamp(record["test_end"]).normalize(),
        )
        for record in records
    )
    if not folds or len({fold.name for fold in folds}) != len(folds):
        raise P0EvidenceError("Temporal outer folds must be non-empty and uniquely named.")
    for fold in folds:
        if not fold.train_end < fold.test_start <= fold.test_end:
            raise P0EvidenceError(f"Invalid temporal boundaries for {fold.name}.")
    ordered = tuple(sorted(folds, key=lambda fold: fold.test_start))
    if ordered != folds:
        raise P0EvidenceError("Temporal outer folds must be ordered by test start.")
    for left, right in zip(folds, folds[1:], strict=False):
        if left.test_end >= right.test_start:
            raise P0EvidenceError("Temporal outer test folds must not overlap.")
    return folds


def make_outer_temporal_split(
    dates: pd.DatetimeIndex,
    fold: OuterFold,
    *,
    purge_days: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if purge_days < 0:
        raise P0EvidenceError("purge_days must be non-negative.")
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    effective_train_end = min(
        fold.train_end,
        fold.test_start - pd.Timedelta(days=int(purge_days) + 1),
    )
    train = np.flatnonzero(normalized <= effective_train_end).astype(int)
    test = np.flatnonzero(
        (normalized >= fold.test_start) & (normalized <= fold.test_end)
    ).astype(int)
    if not len(train) or not len(test):
        raise P0EvidenceError(f"Empty train or test partition for {fold.name}.")
    if np.intersect1d(train, test).size:
        raise P0EvidenceError(f"Outer overlap detected for {fold.name}.")
    train_max = normalized[train].max()
    test_min = normalized[test].min()
    calendar_gap = int((test_min - train_max).days - 1)
    if calendar_gap < purge_days:
        raise P0EvidenceError(f"Outer purge gap is too short for {fold.name}.")
    audit = {
        "fold": fold.name,
        "configured_train_end": fold.train_end,
        "effective_train_end": train_max,
        "test_start": test_min,
        "test_end": normalized[test].max(),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "purged_calendar_days": calendar_gap,
        "outer_overlap_rows": 0,
        "strictly_forward": True,
    }
    return train, test, audit


def make_inner_temporal_folds(
    dates: pd.DatetimeIndex,
    outer_train: np.ndarray,
    *,
    n_splits: int,
    validation_rows: int,
    purge_days: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray, pd.DataFrame]:
    indices = np.asarray(outer_train, dtype=int)
    if n_splits < 2 or validation_rows < 1 or purge_days < 0:
        raise P0EvidenceError("Invalid inner temporal split settings.")
    normalized = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    ordered = indices[np.argsort(normalized[indices].asi8)]
    splitter = TimeSeriesSplit(
        n_splits=int(n_splits),
        test_size=int(validation_rows),
        gap=int(purge_days),
    )
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for fold_id, (fit_position, validation_position) in enumerate(
        splitter.split(ordered), start=1
    ):
        fit_indices = ordered[fit_position]
        validation_indices = ordered[validation_position]
        fit_max = normalized[fit_indices].max()
        validation_min = normalized[validation_indices].min()
        calendar_gap = int((validation_min - fit_max).days - 1)
        if calendar_gap < purge_days:
            raise P0EvidenceError("Inner temporal purge gap is shorter than registered.")
        folds.append((fit_indices.astype(int), validation_indices.astype(int)))
        for role, role_indices in (
            ("inner_train", fit_indices),
            ("inner_validation", validation_indices),
        ):
            records.extend(
                {
                    "inner_fold": fold_id,
                    "row_index": int(index),
                    "Date": normalized[index],
                    "role": role,
                    "purge_days": int(purge_days),
                }
                for index in role_indices
            )
    validation = np.concatenate([fold[1] for fold in folds]).astype(int)
    if len(np.unique(validation)) != len(validation):
        raise P0EvidenceError("Inner temporal validation rows overlap.")
    return folds, validation, pd.DataFrame.from_records(records)


def _seeded_clone(spec: EnhancedModelSpec, seed: int) -> Any:
    estimator = clone(spec.estimator)
    available = estimator.get_params(deep=True)
    seed_parameters = {
        key: int(seed)
        for key in available
        if key.endswith("random_state") or key.endswith("random_seed")
    }
    return estimator.set_params(**seed_parameters)


def fit_temporal_base(
    bundle: RandomFeatureBundle,
    spec: EnhancedModelSpec,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    inner_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    validation_indices: np.ndarray,
    *,
    seed: int,
    progress: Callable[[str], None] | None = None,
) -> TemporalBaseFit:
    X = bundle.matrix(spec.feature_key)
    y = bundle.anchor["actual"].to_numpy(float)
    validation_indices = np.asarray(validation_indices, dtype=int)
    validation_position = {
        int(index): position for position, index in enumerate(validation_indices)
    }
    candidate_predictions: list[np.ndarray] = []
    tuning_records: list[dict[str, Any]] = []
    for candidate_index, parameters in enumerate(spec.candidates):
        oof = np.full(len(validation_indices), np.nan, dtype=float)
        failed = False
        for fold_id, (fit_indices, fold_validation) in enumerate(inner_folds, start=1):
            try:
                estimator = _seeded_clone(spec, seed + fold_id).set_params(**parameters)
                estimator.fit(X.iloc[fit_indices], y[fit_indices])
                prediction = np.asarray(
                    estimator.predict(X.iloc[fold_validation]), dtype=float
                ).reshape(-1)
                if len(prediction) != len(fold_validation) or not np.isfinite(
                    prediction
                ).all():
                    raise RuntimeError("Candidate returned invalid temporal predictions.")
                positions = [validation_position[int(index)] for index in fold_validation]
                oof[positions] = prediction
                rmse = float(mean_squared_error(y[fold_validation], prediction) ** 0.5)
                mae = float(mean_absolute_error(y[fold_validation], prediction))
                status = "completed"
                error = None
            except Exception as exc:
                failed = True
                rmse = np.nan
                mae = np.nan
                status = "failed"
                error = f"{type(exc).__name__}: {exc}"
            tuning_records.append(
                {
                    "target": bundle.target,
                    "model": spec.model_id,
                    "candidate_index": candidate_index,
                    "parameters": str(dict(parameters)),
                    "record_type": "inner_temporal_fold",
                    "inner_fold": fold_id,
                    "inner_RMSE": rmse,
                    "inner_MAE": mae,
                    "status": status,
                    "error": error,
                    "outer_test_accessed": False,
                }
            )
        if failed or not np.isfinite(oof).all():
            candidate_predictions.append(np.full_like(oof, np.nan))
            continue
        pooled_rmse = float(mean_squared_error(y[validation_indices], oof) ** 0.5)
        pooled_mae = float(mean_absolute_error(y[validation_indices], oof))
        tuning_records.append(
            {
                "target": bundle.target,
                "model": spec.model_id,
                "candidate_index": candidate_index,
                "parameters": str(dict(parameters)),
                "record_type": "pooled_inner_temporal",
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
        if record["record_type"] == "pooled_inner_temporal"
        and record["status"] == "completed"
    ]
    if not completed:
        raise P0EvidenceError(f"All temporal candidates failed for {spec.model_id}.")
    best_rmse = min(float(record["inner_RMSE"]) for record in completed)
    near = [
        record
        for record in completed
        if float(record["inner_RMSE"]) <= best_rmse * 1.01
    ]
    selected = min(
        near,
        key=lambda record: (float(record["inner_MAE"]), int(record["candidate_index"])),
    )
    selected_index = int(selected["candidate_index"])
    selected_parameters = dict(spec.candidates[selected_index])
    if progress is not None:
        progress(f"{bundle.target}/{spec.model_id}: candidate={selected_index}")
    final = _seeded_clone(spec, seed).set_params(**selected_parameters)
    final.fit(X.iloc[outer_train], y[np.asarray(outer_train, dtype=int)])
    test_prediction = np.asarray(final.predict(X.iloc[outer_test]), dtype=float).reshape(-1)
    if len(test_prediction) != len(outer_test) or not np.isfinite(test_prediction).all():
        raise P0EvidenceError(f"Invalid outer temporal predictions from {spec.model_id}.")
    return TemporalBaseFit(
        model=spec.model_id,
        selected_candidate=selected_index,
        selected_parameters=selected_parameters,
        validation_indices=validation_indices.copy(),
        validation_prediction=candidate_predictions[selected_index],
        test_prediction=test_prediction,
        tuning_records=tuple(tuning_records),
    )


def run_temporal_validation(
    feature_set: PairedFeatureSet,
    registries: Mapping[str, Mapping[str, EnhancedModelSpec]],
    outer_folds: Sequence[OuterFold],
    *,
    purge_days: int,
    inner_splits: int,
    inner_validation_rows: int,
    seed: int,
    huber_epsilon: float,
    huber_alpha: float,
    progress: Callable[[str], None] | None = None,
) -> TemporalValidationResult:
    dates = pd.DatetimeIndex(pd.to_datetime(feature_set.common_dates)).normalize()
    prediction_frames: list[pd.DataFrame] = []
    metric_records: list[dict[str, Any]] = []
    selected_records: list[dict[str, Any]] = []
    tuning_records: list[dict[str, Any]] = []
    weight_records: list[dict[str, Any]] = []
    outer_records: list[dict[str, Any]] = []
    inner_frames: list[pd.DataFrame] = []
    leakage_records: list[dict[str, Any]] = []
    seen_test_indices: set[int] = set()

    for outer_id, outer_fold in enumerate(outer_folds, start=1):
        outer_train, outer_test, audit = make_outer_temporal_split(
            dates, outer_fold, purge_days=purge_days
        )
        if seen_test_indices.intersection(int(index) for index in outer_test):
            raise P0EvidenceError("Outer temporal test rows overlap across folds.")
        seen_test_indices.update(int(index) for index in outer_test)
        inner_folds, validation_indices, inner_table = make_inner_temporal_folds(
            dates,
            outer_train,
            n_splits=inner_splits,
            validation_rows=inner_validation_rows,
            purge_days=purge_days,
        )
        inner_table.insert(0, "outer_fold", outer_fold.name)
        inner_frames.append(inner_table)
        for role, indices in (("outer_train", outer_train), ("outer_test", outer_test)):
            outer_records.extend(
                {
                    "outer_fold": outer_fold.name,
                    "row_index": int(index),
                    "Date": dates[index],
                    "role": role,
                }
                for index in indices
            )
        fold_seed = int(seed) + outer_id * 10_000
        if progress is not None:
            progress(
                f"fold={outer_fold.name}: train={len(outer_train)} test={len(outer_test)}"
            )

        dec_bundle = feature_set.bundles["DEC"]
        dec_fit = fit_temporal_base(
            dec_bundle,
            registries["DEC"]["ExtraTrees"],
            outer_train,
            outer_test,
            inner_folds,
            validation_indices,
            seed=fold_seed,
            progress=progress,
        )
        tuning_records.extend(
            {"outer_fold": outer_fold.name, **record}
            for record in dec_fit.tuning_records
        )
        selected_records.append(
            {
                "outer_fold": outer_fold.name,
                "target": "DEC",
                "model": "ExtraTrees",
                "component_model": "ExtraTrees",
                "selected_candidate": dec_fit.selected_candidate,
                "selected_parameters": str(dict(dec_fit.selected_parameters)),
                "selection_scope": "inner_expanding_window_only",
            }
        )

        tn_bundle = feature_set.bundles["TN_out"]
        base_validation = pd.DataFrame(index=validation_indices, columns=BASE_MODELS, dtype=float)
        base_test = pd.DataFrame(index=outer_test, columns=BASE_MODELS, dtype=float)
        for model in BASE_MODELS:
            fitted = fit_temporal_base(
                tn_bundle,
                registries["TN_out"][model],
                outer_train,
                outer_test,
                inner_folds,
                validation_indices,
                seed=fold_seed,
                progress=progress,
            )
            base_validation.loc[:, model] = fitted.validation_prediction
            base_test.loc[:, model] = fitted.test_prediction
            tuning_records.extend(
                {"outer_fold": outer_fold.name, **record}
                for record in fitted.tuning_records
            )
            selected_records.append(
                {
                    "outer_fold": outer_fold.name,
                    "target": "TN_out",
                    "model": "Ensemble_Huber",
                    "component_model": model,
                    "selected_candidate": fitted.selected_candidate,
                    "selected_parameters": str(dict(fitted.selected_parameters)),
                    "selection_scope": "inner_expanding_window_only",
                }
            )
        if base_validation.isna().any().any() or base_test.isna().any().any():
            raise P0EvidenceError("Temporal Huber base matrix is incomplete.")
        scaler = StandardScaler().fit(base_validation)
        meta_model = HuberRegressor(
            epsilon=float(huber_epsilon),
            alpha=float(huber_alpha),
            max_iter=2_000,
            tol=1e-7,
        ).fit(
            scaler.transform(base_validation),
            tn_bundle.anchor.iloc[validation_indices]["actual"].to_numpy(float),
        )
        tn_prediction = np.asarray(
            meta_model.predict(scaler.transform(base_test)), dtype=float
        ).reshape(-1)
        effective_weights = np.asarray(meta_model.coef_, dtype=float) / scaler.scale_
        effective_intercept = float(
            meta_model.intercept_ - np.sum(meta_model.coef_ * scaler.mean_ / scaler.scale_)
        )
        weight_records.extend(
            {
                "outer_fold": outer_fold.name,
                "ensemble_model": "Ensemble_Huber",
                "component_model": model,
                "effective_weight": float(weight),
                "effective_intercept": effective_intercept,
                "meta_epsilon": float(huber_epsilon),
                "meta_alpha": float(huber_alpha),
            }
            for model, weight in zip(BASE_MODELS, effective_weights, strict=True)
        )

        for target, model, prediction in (
            ("DEC", "ExtraTrees", dec_fit.test_prediction),
            ("TN_out", "Ensemble_Huber", tn_prediction),
        ):
            bundle = feature_set.bundles[target]
            actual = bundle.anchor.iloc[outer_test]["actual"].to_numpy(float)
            frame = pd.DataFrame(
                {
                    "outer_fold": outer_fold.name,
                    "target": target,
                    "model": model,
                    "Date": dates[outer_test],
                    "actual": actual,
                    "prediction": prediction,
                    "analysis_role": "post_hoc_rolling_origin_sensitivity",
                    "independent_test": False,
                    "future_prediction_claim_authorized": False,
                }
            )
            prediction_frames.append(frame)
            metric_records.append(
                {
                    "outer_fold": outer_fold.name,
                    "target": target,
                    "model": model,
                    "n_train": int(len(outer_train)),
                    "n_test": int(len(outer_test)),
                    **regression_metrics(actual, prediction),
                }
            )
        leakage_records.append(
            {
                **audit,
                "inner_splits": int(inner_splits),
                "inner_validation_rows_per_split": int(inner_validation_rows),
                "outer_test_accessed_for_hyperparameter_or_meta_fit": False,
                "preprocessing_fit_scope": "inner_or_outer_training_only",
                "target_history_columns_used": 0,
                "effluent_target_features_used": 0,
                "independent_test": False,
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    pooled_records: list[dict[str, Any]] = []
    for (target, model), group in predictions.groupby(["target", "model"], sort=False):
        pooled_records.append(
            {
                "target": target,
                "model": model,
                "n_test": int(len(group)),
                "test_start": group["Date"].min(),
                "test_end": group["Date"].max(),
                **regression_metrics(
                    group["actual"].to_numpy(float),
                    group["prediction"].to_numpy(float),
                ),
                "analysis_role": "pooled_disjoint_rolling_origin_folds_post_hoc",
                "independent_test": False,
            }
        )
    return TemporalValidationResult(
        predictions=predictions,
        metrics_by_fold=pd.DataFrame.from_records(metric_records),
        pooled_metrics=pd.DataFrame.from_records(pooled_records),
        selected_hyperparameters=pd.DataFrame.from_records(selected_records),
        tuning_trials=pd.DataFrame.from_records(tuning_records),
        ensemble_weights=pd.DataFrame.from_records(weight_records),
        outer_assignments=pd.DataFrame.from_records(outer_records),
        inner_assignments=pd.concat(inner_frames, ignore_index=True),
        leakage_audit=pd.DataFrame.from_records(leakage_records),
    )


def build_joint_residual_library(
    outer_predictions: pd.DataFrame,
    *,
    training_window: str,
) -> pd.DataFrame:
    required = {"target", "training_window", "model", "Date", "actual", "prediction"}
    missing = required.difference(outer_predictions.columns)
    if missing:
        raise P0EvidenceError(f"Outer prediction table lacks columns: {sorted(missing)}")
    frame = outer_predictions.copy()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.normalize()
    selected = frame.loc[
        frame["training_window"].eq(training_window)
        & (
            (frame["target"].eq("TN_out") & frame["model"].eq("Ensemble_Huber"))
            | (frame["target"].eq("DEC") & frame["model"].eq("ExtraTrees"))
        )
    ].copy()
    if selected.empty:
        raise P0EvidenceError("No frozen final-model residual rows were found.")
    actual_nunique = selected.groupby(["target", "Date"])["actual"].nunique()
    if not actual_nunique.eq(1).all():
        raise P0EvidenceError("Repeated outer predictions disagree on actual values.")
    aggregated = (
        selected.groupby(["target", "Date"], as_index=False)
        .agg(actual=("actual", "first"), prediction=("prediction", "median"), repeats=("prediction", "size"))
    )
    aggregated["residual"] = aggregated["actual"] - aggregated["prediction"]
    wide = aggregated.pivot(index="Date", columns="target", values="residual")
    repeats = aggregated.pivot(index="Date", columns="target", values="repeats")
    wide = wide.dropna(subset=["TN_out", "DEC"]).sort_index()
    if len(wide) < 30:
        raise P0EvidenceError("Joint residual library is too small.")
    result = wide.rename(columns={"TN_out": "residual_TN", "DEC": "residual_DEC"}).reset_index()
    repeat_rows = repeats.loc[wide.index]
    result["TN_repeats"] = repeat_rows["TN_out"].to_numpy(int)
    result["DEC_repeats"] = repeat_rows["DEC"].to_numpy(int)
    result["residual_role"] = "date_aggregated_overlapping_outer_predictions_not_external"
    return result


def propagate_proxy_uncertainty(
    failure_cases: pd.DataFrame,
    residual_library: pd.DataFrame,
    *,
    correlations: Sequence[float],
    primary_correlation: float,
    replicates: int,
    seed: int,
    interval_level: float,
) -> ProxyUncertaintyResult:
    required = {
        "method",
        "training_seed",
        "preference_TN",
        "episode_id",
        "start_date",
        "end_date",
        "delta_TN_vs_RandomFeasible",
        "delta_DEC_vs_RandomFeasible",
    }
    missing = required.difference(failure_cases.columns)
    if missing:
        raise P0EvidenceError(f"Failure-case table lacks columns: {sorted(missing)}")
    correlations = tuple(float(value) for value in correlations)
    if not correlations or any(value < 0 or value >= 1 for value in correlations):
        raise P0EvidenceError("Correlations must lie in [0, 1).")
    if float(primary_correlation) not in correlations:
        raise P0EvidenceError("Primary correlation must be included in the sensitivity grid.")
    if replicates < 1_000 or not 0.5 < interval_level < 1.0:
        raise P0EvidenceError("Uncertainty replication or interval setting is invalid.")
    frame = failure_cases.loc[failure_cases["method"].isin(RL_METHODS)].copy()
    if set(frame["method"]) != set(RL_METHODS):
        raise P0EvidenceError("All three learned RL methods are required.")
    frame["start_date"] = pd.to_datetime(frame["start_date"]).dt.normalize()
    frame["end_date"] = pd.to_datetime(frame["end_date"]).dt.normalize()
    residuals = residual_library[["residual_TN", "residual_DEC"]].to_numpy(float)
    if not np.isfinite(residuals).all():
        raise P0EvidenceError("Residual library contains invalid values.")
    residuals = residuals - residuals.mean(axis=0, keepdims=True)
    lower_q = (1.0 - float(interval_level)) / 2.0
    upper_q = 1.0 - lower_q
    rng = np.random.default_rng(int(seed))
    episode_records: list[dict[str, Any]] = []
    method_records: list[dict[str, Any]] = []

    for correlation in correlations:
        differential_scale = float(np.sqrt(1.0 - correlation))
        for method in RL_METHODS:
            subset = frame.loc[frame["method"].eq(method)].reset_index(drop=True)
            observed = subset[
                ["delta_TN_vs_RandomFeasible", "delta_DEC_vs_RandomFeasible"]
            ].to_numpy(float)
            sample_a = rng.integers(0, len(residuals), size=(len(subset), replicates))
            sample_b = rng.integers(0, len(residuals), size=(len(subset), replicates))
            error = differential_scale * (residuals[sample_a] - residuals[sample_b])
            draws = observed[:, None, :] + error
            low = np.quantile(draws, lower_q, axis=1)
            high = np.quantile(draws, upper_q, axis=1)
            probability_tn_lower = np.mean(draws[:, :, 0] < 0.0, axis=1)
            probability_dec_lower = np.mean(draws[:, :, 1] < 0.0, axis=1)
            probability_rl_dominates = np.mean(
                (draws[:, :, 0] < 0.0) & (draws[:, :, 1] < 0.0), axis=1
            )
            probability_rf_dominates = np.mean(
                (draws[:, :, 0] > 0.0) & (draws[:, :, 1] > 0.0), axis=1
            )
            for position, row in subset.iterrows():
                episode_records.append(
                    {
                        "correlation": correlation,
                        "is_primary_correlation": correlation == float(primary_correlation),
                        "method": method,
                        "training_seed": int(row["training_seed"]),
                        "preference_TN": float(row["preference_TN"]),
                        "episode_id": int(row["episode_id"]),
                        "start_date": row["start_date"],
                        "end_date": row["end_date"],
                        "observed_delta_TN": observed[position, 0],
                        "observed_delta_DEC": observed[position, 1],
                        "delta_TN_interval_low": low[position, 0],
                        "delta_TN_interval_high": high[position, 0],
                        "delta_DEC_interval_low": low[position, 1],
                        "delta_DEC_interval_high": high[position, 1],
                        "probability_RL_lower_TN": probability_tn_lower[position],
                        "probability_RL_lower_DEC": probability_dec_lower[position],
                        "probability_RL_dominates_RF": probability_rl_dominates[position],
                        "probability_RF_dominates_RL": probability_rf_dominates[position],
                        "TN_sign_resolved_95": bool(
                            high[position, 0] < 0 or low[position, 0] > 0
                        ),
                        "DEC_sign_resolved_95": bool(
                            high[position, 1] < 0 or low[position, 1] > 0
                        ),
                        "calibrated_coverage_claim_authorized": False,
                    }
                )

            systematic_a = rng.integers(0, len(residuals), size=replicates)
            systematic_b = rng.integers(0, len(residuals), size=replicates)
            mean_draws = observed.mean(axis=0) + differential_scale * (
                residuals[systematic_a] - residuals[systematic_b]
            )
            primary_episode_rows = [
                record
                for record in episode_records
                if record["correlation"] == correlation and record["method"] == method
            ]
            method_records.append(
                {
                    "correlation": correlation,
                    "is_primary_correlation": correlation == float(primary_correlation),
                    "method": method,
                    "episodes": int(len(subset)),
                    "observed_mean_delta_TN": float(observed[:, 0].mean()),
                    "observed_mean_delta_DEC": float(observed[:, 1].mean()),
                    "systematic_delta_TN_interval_low": float(
                        np.quantile(mean_draws[:, 0], lower_q)
                    ),
                    "systematic_delta_TN_interval_high": float(
                        np.quantile(mean_draws[:, 0], upper_q)
                    ),
                    "systematic_delta_DEC_interval_low": float(
                        np.quantile(mean_draws[:, 1], lower_q)
                    ),
                    "systematic_delta_DEC_interval_high": float(
                        np.quantile(mean_draws[:, 1], upper_q)
                    ),
                    "probability_mean_RL_dominates_RF": float(
                        np.mean((mean_draws[:, 0] < 0) & (mean_draws[:, 1] < 0))
                    ),
                    "probability_mean_RF_dominates_RL": float(
                        np.mean((mean_draws[:, 0] > 0) & (mean_draws[:, 1] > 0))
                    ),
                    "episodes_TN_sign_resolved_95": int(
                        sum(record["TN_sign_resolved_95"] for record in primary_episode_rows)
                    ),
                    "episodes_DEC_sign_resolved_95": int(
                        sum(record["DEC_sign_resolved_95"] for record in primary_episode_rows)
                    ),
                    "uncertainty_role": (
                        "empirical_outer_residual_sensitivity_not_calibrated_counterfactual_CI"
                    ),
                    "plant_control_claim_authorized": False,
                }
            )

    scale_records = []
    for target, column, unit in (
        ("TN_out", "residual_TN", "mg/L"),
        ("DEC", "residual_DEC", "kWh/d"),
    ):
        values = residual_library[column].to_numpy(float)
        scale_records.append(
            {
                "target": target,
                "unit": unit,
                "unique_dates": int(len(values)),
                "RMSE": float(np.sqrt(np.mean(values**2))),
                "MAE": float(np.mean(np.abs(values))),
                "residual_sd": float(np.std(values, ddof=1)),
                "residual_q025": float(np.quantile(values, 0.025)),
                "residual_q975": float(np.quantile(values, 0.975)),
                "evidence_role": "overlapping_outer_predictions_aggregated_by_unique_date",
            }
        )
    return ProxyUncertaintyResult(
        residual_library=residual_library.copy(),
        residual_scale=pd.DataFrame.from_records(scale_records),
        episode_intervals=pd.DataFrame.from_records(episode_records),
        method_summary=pd.DataFrame.from_records(method_records),
    )
