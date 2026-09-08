"""Explainability for the frozen DEC and TN_out proxy choices."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone

from .explainability import compute_ale_1d, compute_tree_shap, fit_ale_grid
from .final_proxy import FinalProxyBundle
from .metrics import regression_metrics
from .paired_random_windows import BASE_MODELS, FEATURE_KEY, OUTER_SEEDS, PairedFeatureSet


class FinalXAIError(ValueError):
    """Raised when the frozen final-XAI contract cannot be reproduced."""


def _seeded_clone(estimator: Any, seed: int) -> Any:
    copied = clone(estimator)
    available = copied.get_params(deep=True)
    seed_parameters = {
        key: int(seed)
        for key in available
        if key.endswith("random_state") or key.endswith("random_seed")
    }
    return copied.set_params(**seed_parameters)


def model_feature_groups(feature_names: Sequence[str]) -> dict[str, list[str]]:
    """Map every model input to one compact physical/temporal group."""

    groups = {
        "Calendar/trend": [],
        "Influent/load": [],
        "Operation proxy": [],
        "Process state": [],
    }
    for feature in (str(value) for value in feature_names):
        if feature in {"doy_sin", "doy_cos", "time_index_days"}:
            groups["Calendar/trend"].append(feature)
        elif feature in {"PPA", "DO"}:
            groups["Operation proxy"].append(feature)
        elif feature == "MLSS":
            groups["Process state"].append(feature)
        else:
            groups["Influent/load"].append(feature)
    assigned = [feature for members in groups.values() for feature in members]
    if sorted(assigned) != sorted(str(value) for value in feature_names):
        raise FinalXAIError("The physical-group mapping is incomplete.")
    return {name: members for name, members in groups.items() if members}


@dataclass
class EffectiveLinearEnsemble:
    """Outer-split Huber ensemble expressed in raw component-prediction units."""

    feature_names: tuple[str, ...]
    component_order: tuple[str, ...]
    base_models: Mapping[str, Any]
    weights: np.ndarray
    intercept: float

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names)
        frame = X.loc[:, list(self.feature_names)]
        components = np.column_stack(
            [
                np.asarray(self.base_models[name].predict(frame), dtype=float).reshape(-1)
                for name in self.component_order
            ]
        )
        prediction = self.intercept + components @ np.asarray(self.weights, dtype=float)
        if not np.isfinite(prediction).all():
            raise FinalXAIError("The reconstructed Huber ensemble returned invalid values.")
        return prediction


@dataclass(frozen=True)
class CrossFittedXAIResult:
    reproduction_audit: pd.DataFrame
    permutation_repeats: pd.DataFrame
    permutation_summary: pd.DataFrame
    dec_shap_values: pd.DataFrame
    dec_shap_audit: pd.DataFrame


@dataclass(frozen=True)
class DeploymentXAIResult:
    tn_shap_values: pd.DataFrame
    tn_shap_audit: pd.DataFrame
    tn_shap_global: pd.DataFrame
    dec_shap_global: pd.DataFrame
    ale_curves: pd.DataFrame
    local_cases: pd.DataFrame


def _outer_indices(
    bundle: Any, assignments: pd.DataFrame, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    dates = pd.DatetimeIndex(pd.to_datetime(bundle.anchor["Date"])).normalize()
    assigned = assignments.loc[assignments["seed"].eq(seed)].copy()
    assigned["Date"] = pd.to_datetime(assigned["Date"]).dt.normalize()
    test_dates = set(assigned.loc[assigned["role"].eq("outer_test"), "Date"])
    train_2025 = set(assigned.loc[assigned["role"].eq("outer_train"), "Date"])
    outer_test = np.flatnonzero(dates.isin(test_dates))
    outer_train = np.flatnonzero((dates < pd.Timestamp("2025-01-01")) | dates.isin(train_2025))
    if len(outer_test) != 70 or len(outer_train) != 1006:
        raise FinalXAIError(
            f"Unexpected three-year outer sizes for seed {seed}: "
            f"train={len(outer_train)}, test={len(outer_test)}"
        )
    if np.intersect1d(outer_train, outer_test).size:
        raise FinalXAIError("Outer train and test rows overlap during XAI reconstruction.")
    return outer_train.astype(int), outer_test.astype(int)


def _selected_parameters(
    selected: pd.DataFrame,
    registry: Mapping[str, Any],
    *,
    target: str,
    seed: int,
    model: str,
) -> dict[str, Any]:
    rows = selected.loc[
        selected["target"].eq(target)
        & selected["training_window"].eq("2023_2025")
        & selected["seed"].eq(seed)
        & selected["model"].eq(model)
    ]
    if len(rows) != 1:
        raise FinalXAIError(f"Expected one parameter lock for {target}/{model}/seed={seed}.")
    row = rows.iloc[0]
    candidate_index = int(float(row["selected_candidate"]))
    candidates = registry[model].candidates
    if candidate_index < 0 or candidate_index >= len(candidates):
        raise FinalXAIError(
            f"Frozen candidate {candidate_index} is invalid for {target}/{model}/seed={seed}."
        )
    parameters = dict(candidates[candidate_index])
    recorded = dict(json.loads(str(row["selected_parameters"])))
    serialized = dict(
        json.loads(json.dumps(parameters, ensure_ascii=False, default=str))
    )
    if serialized != recorded:
        raise FinalXAIError(
            f"Registry candidate differs from the frozen parameter record for "
            f"{target}/{model}/seed={seed}."
        )
    return parameters


def _hubert_outer_model(
    feature_set: PairedFeatureSet,
    registry: Mapping[str, Any],
    selected: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    seed: int,
    outer_train: np.ndarray,
) -> EffectiveLinearEnsemble:
    bundle = feature_set.bundles["TN_out"]
    X = bundle.matrix(FEATURE_KEY)
    y = bundle.anchor["actual"].to_numpy(float)
    fitted: dict[str, Any] = {}
    for model in BASE_MODELS:
        parameters = _selected_parameters(
            selected,
            registry,
            target="TN_out",
            seed=seed,
            model=model,
        )
        estimator = _seeded_clone(registry[model].estimator, seed).set_params(**parameters)
        estimator.fit(X.iloc[outer_train], y[outer_train])
        fitted[model] = estimator
    rows = weights.loc[
        weights["target"].eq("TN_out")
        & weights["training_window"].eq("2023_2025")
        & weights["seed"].eq(seed)
        & weights["ensemble_model"].eq("Ensemble_Huber")
    ].copy()
    rows["component_model"] = pd.Categorical(
        rows["component_model"], categories=BASE_MODELS, ordered=True
    )
    rows = rows.sort_values("component_model")
    if tuple(rows["component_model"].astype(str)) != tuple(BASE_MODELS):
        raise FinalXAIError("Huber component weights are incomplete or out of order.")
    intercepts = pd.to_numeric(rows["intercept"], errors="raise").unique()
    if len(intercepts) != 1:
        raise FinalXAIError("Huber component rows do not share one intercept.")
    return EffectiveLinearEnsemble(
        feature_names=tuple(bundle.feature_names[FEATURE_KEY]),
        component_order=tuple(BASE_MODELS),
        base_models=fitted,
        weights=pd.to_numeric(rows["weight"], errors="raise").to_numpy(float),
        intercept=float(intercepts[0]),
    )


def _dec_outer_model(
    feature_set: PairedFeatureSet,
    registry: Mapping[str, Any],
    selected: pd.DataFrame,
    *,
    seed: int,
    outer_train: np.ndarray,
) -> Any:
    bundle = feature_set.bundles["DEC"]
    parameters = _selected_parameters(
        selected,
        registry,
        target="DEC",
        seed=seed,
        model="ExtraTrees",
    )
    estimator = _seeded_clone(registry["ExtraTrees"].estimator, seed).set_params(**parameters)
    estimator.fit(
        bundle.matrix(FEATURE_KEY).iloc[outer_train],
        bundle.anchor.iloc[outer_train]["actual"].to_numpy(float),
    )
    return estimator


def _permutation_rows(
    estimator: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    *,
    target: str,
    seed: int,
    repeats: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    baseline_prediction = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
    baseline = regression_metrics(y_test, baseline_prediction)
    rng = np.random.default_rng(int(random_seed) + int(seed) + (0 if target == "TN_out" else 10_000))
    records: list[dict[str, Any]] = []
    for feature in X_test.columns:
        for repeat in range(1, int(repeats) + 1):
            permuted = X_test.copy()
            permuted.loc[:, feature] = rng.permutation(permuted[feature].to_numpy())
            prediction = np.asarray(estimator.predict(permuted), dtype=float).reshape(-1)
            metrics = regression_metrics(y_test, prediction)
            records.append(
                {
                    "target": target,
                    "seed": int(seed),
                    "feature": feature,
                    "repeat": repeat,
                    "baseline_RMSE": baseline["RMSE"],
                    "permuted_RMSE": metrics["RMSE"],
                    "delta_RMSE": metrics["RMSE"] - baseline["RMSE"],
                    "baseline_MAE": baseline["MAE"],
                    "permuted_MAE": metrics["MAE"],
                    "delta_MAE": metrics["MAE"] - baseline["MAE"],
                    "interpretation": "predictive_association_not_causal_effect",
                }
            )
    return records


def run_crossfitted_xai(
    feature_set: PairedFeatureSet,
    registries: Mapping[str, Mapping[str, Any]],
    parent_run: Path,
    *,
    permutation_repeats: int = 20,
    random_seed: int = 20260823,
    reproduction_tolerance: float = 1e-8,
) -> CrossFittedXAIResult:
    """Rebuild the five frozen outer models and explain their held-out rows."""

    selected = pd.read_csv(parent_run / "selected_hyperparameters.csv", encoding="utf-8-sig")
    weights = pd.read_csv(parent_run / "ensemble_weights.csv", encoding="utf-8-sig")
    assignments = pd.read_csv(
        parent_run / "outer_assignments.csv", encoding="utf-8-sig", parse_dates=["Date"]
    )
    frozen = pd.read_csv(
        parent_run / "outer_predictions.csv", encoding="utf-8-sig", parse_dates=["Date"]
    )
    audit_records: list[dict[str, Any]] = []
    permutation_records: list[dict[str, Any]] = []
    dec_shap_records: list[dict[str, Any]] = []
    shap_audit_records: list[dict[str, Any]] = []
    for seed in OUTER_SEEDS:
        for target in ("TN_out", "DEC"):
            bundle = feature_set.bundles[target]
            outer_train, outer_test = _outer_indices(bundle, assignments, seed)
            X = bundle.matrix(FEATURE_KEY)
            X_test = X.iloc[outer_test].reset_index(drop=True)
            y_test = bundle.anchor.iloc[outer_test]["actual"].to_numpy(float)
            if target == "TN_out":
                estimator = _hubert_outer_model(
                    feature_set,
                    registries[target],
                    selected,
                    weights,
                    seed=seed,
                    outer_train=outer_train,
                )
                frozen_model = "Ensemble_Huber"
            else:
                estimator = _dec_outer_model(
                    feature_set,
                    registries[target],
                    selected,
                    seed=seed,
                    outer_train=outer_train,
                )
                frozen_model = "ExtraTrees"
            prediction = np.asarray(estimator.predict(X_test), dtype=float).reshape(-1)
            dates = pd.to_datetime(bundle.anchor.iloc[outer_test]["Date"]).reset_index(drop=True)
            expected = frozen.loc[
                frozen["target"].eq(target)
                & frozen["training_window"].eq("2023_2025")
                & frozen["seed"].eq(seed)
                & frozen["model"].eq(frozen_model),
                ["Date", "actual", "prediction"],
            ].sort_values("Date")
            observed = pd.DataFrame(
                {"Date": dates, "actual": y_test, "prediction": prediction}
            ).sort_values("Date")
            if len(expected) != len(observed) or not expected["Date"].reset_index(drop=True).equals(
                observed["Date"].reset_index(drop=True)
            ):
                raise FinalXAIError("Frozen and rebuilt outer prediction dates differ.")
            maximum_error = float(
                np.max(
                    np.abs(
                        expected["prediction"].to_numpy(float)
                        - observed["prediction"].to_numpy(float)
                    )
                )
            )
            if maximum_error > reproduction_tolerance:
                raise FinalXAIError(
                    f"Outer prediction reproduction failed for {target}/seed={seed}: "
                    f"{maximum_error:.3g}"
                )
            audit_records.append(
                {
                    "target": target,
                    "model": frozen_model,
                    "seed": seed,
                    "outer_train_rows": len(outer_train),
                    "outer_test_rows": len(outer_test),
                    "max_abs_prediction_reproduction_error": maximum_error,
                    "tolerance": reproduction_tolerance,
                    "passed": True,
                }
            )
            permutation_records.extend(
                _permutation_rows(
                    estimator,
                    X_test,
                    y_test,
                    target=target,
                    seed=seed,
                    repeats=permutation_repeats,
                    random_seed=random_seed,
                )
            )
            if target == "DEC":
                feature_names = tuple(bundle.feature_names[FEATURE_KEY])
                result = compute_tree_shap(
                    estimator,
                    X_test,
                    feature_names,
                    model_feature_groups(feature_names),
                    additivity_tolerance=1e-8,
                )
                metadata = result.transformed_design.metadata
                for feature in feature_names:
                    columns = metadata.index[metadata["source_feature"].eq(feature)].to_numpy(int)
                    feature_values = X_test[feature].to_numpy(float)
                    shap_values = result.values[:, columns].sum(axis=1)
                    for row_index, date in enumerate(dates):
                        dec_shap_records.append(
                            {
                                "target": "DEC",
                                "model": "ExtraTrees",
                                "seed": seed,
                                "Date": date,
                                "feature": feature,
                                "feature_value": feature_values[row_index],
                                "shap_value": shap_values[row_index],
                                "base_value": result.base_values[row_index],
                                "prediction": result.predictions[row_index],
                                "explanation_scope": "crossfitted_outer_random_holdout",
                            }
                        )
                shap_audit_records.append(
                    {
                        "target": "DEC",
                        "seed": seed,
                        "rows": len(X_test),
                        "max_additivity_error": result.max_additivity_error,
                        "raw_feature_conservation_error": float(
                            np.max(
                                np.abs(
                                    result.values.sum(axis=1)
                                    - np.column_stack(
                                        [
                                            result.values[
                                                :,
                                                metadata.index[
                                                    metadata["source_feature"].eq(feature)
                                                ].to_numpy(int),
                                            ].sum(axis=1)
                                            for feature in feature_names
                                        ]
                                    ).sum(axis=1)
                                )
                            )
                        ),
                        "passed": True,
                    }
                )
    permutation = pd.DataFrame.from_records(permutation_records)
    summary = (
        permutation.groupby(["target", "feature"], observed=True)
        .agg(
            mean_delta_RMSE=("delta_RMSE", "mean"),
            sd_delta_RMSE=("delta_RMSE", "std"),
            q025_delta_RMSE=("delta_RMSE", lambda values: float(np.quantile(values, 0.025))),
            q975_delta_RMSE=("delta_RMSE", lambda values: float(np.quantile(values, 0.975))),
            mean_delta_MAE=("delta_MAE", "mean"),
            n_outer_seeds=("seed", "nunique"),
            n_permutations=("delta_RMSE", "size"),
        )
        .reset_index()
        .sort_values(["target", "mean_delta_RMSE"], ascending=[True, False])
    )
    return CrossFittedXAIResult(
        reproduction_audit=pd.DataFrame.from_records(audit_records),
        permutation_repeats=permutation,
        permutation_summary=summary,
        dec_shap_values=pd.DataFrame.from_records(dec_shap_records),
        dec_shap_audit=pd.DataFrame.from_records(shap_audit_records),
    )


def _evenly_spaced_indices(n_rows: int, requested: int) -> np.ndarray:
    if n_rows < 1 or requested < 1:
        raise FinalXAIError("Explanation sample sizes must be positive.")
    return np.unique(np.linspace(0, n_rows - 1, min(n_rows, requested), dtype=int))


def _local_cases(parent_predictions: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for target, model in (("TN_out", "Ensemble_Huber"), ("DEC", "ExtraTrees")):
        selected = parent_predictions.loc[
            parent_predictions["target"].eq(target)
            & parent_predictions["training_window"].eq("2023_2025")
            & parent_predictions["model"].eq(model),
            ["Date", "actual", "prediction"],
        ].copy()
        by_date = selected.groupby("Date", as_index=False).agg(
            actual=("actual", "first"), prediction=("prediction", "mean")
        )
        by_date["absolute_error"] = (by_date["actual"] - by_date["prediction"]).abs()
        low, high = by_date["actual"].quantile([0.40, 0.60])
        middle = by_date.loc[by_date["actual"].between(low, high)]
        choices = {
            "typical_accurate": middle.sort_values(["absolute_error", "Date"]).iloc[0],
            "highest_target": by_date.sort_values(
                ["actual", "Date"], ascending=[False, True]
            ).iloc[0],
            "largest_error": by_date.sort_values(
                ["absolute_error", "Date"], ascending=[False, True]
            ).iloc[0],
        }
        for case_type, row in choices.items():
            records.append(
                {
                    "target": target,
                    "model": model,
                    "case_type": case_type,
                    "Date": pd.Timestamp(row["Date"]),
                    "actual": float(row["actual"]),
                    "mean_outer_prediction": float(row["prediction"]),
                    "absolute_error": float(row["absolute_error"]),
                    "selection_rule": "predefined_from_frozen_outer_predictions",
                }
            )
    return pd.DataFrame.from_records(records)


def run_deployment_xai(
    final_bundle: FinalProxyBundle,
    feature_set: PairedFeatureSet,
    crossfitted: CrossFittedXAIResult,
    parent_run: Path,
    *,
    tn_shap_rows: int = 120,
    tn_background_rows: int = 48,
    tn_shap_permutations: int = 3,
    tn_shap_additivity_tolerance: float = 1e-4,
    ale_features: Sequence[str] = (
        "PPA",
        "DO",
        "MLSS",
    ),
    ale_quantile_bins: int = 8,
    ale_support_lower_quantile: float = 0.05,
    ale_support_upper_quantile: float = 0.95,
    random_seed: int = 20260823,
) -> DeploymentXAIResult:
    """Explain the full-data artifacts while retaining explicit analysis labels."""

    if (
        not np.isfinite(tn_shap_additivity_tolerance)
        or tn_shap_additivity_tolerance <= 0
    ):
        raise FinalXAIError("TN SHAP additivity tolerance must be positive and finite.")

    tn_bundle = feature_set.bundles["TN_out"]
    X_tn = tn_bundle.matrix(FEATURE_KEY).reset_index(drop=True)
    dates = pd.to_datetime(tn_bundle.anchor["Date"]).reset_index(drop=True)
    explain_indices = _evenly_spaced_indices(len(X_tn), int(tn_shap_rows))
    background_indices = _evenly_spaced_indices(len(X_tn), int(tn_background_rows))
    explain_frame = X_tn.iloc[explain_indices].reset_index(drop=True)
    background = X_tn.iloc[background_indices].reset_index(drop=True)
    feature_names = tuple(tn_bundle.feature_names[FEATURE_KEY])

    def predict_array(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=feature_names)
        return final_bundle.predict_tn(frame)

    masker = shap.maskers.Independent(background, max_samples=len(background))
    explainer = shap.PermutationExplainer(
        predict_array,
        masker,
        feature_names=list(feature_names),
        seed=int(random_seed),
    )
    minimum_evals = 2 * len(feature_names) + 1
    max_evals = int(tn_shap_permutations) * minimum_evals
    explanation = explainer(
        explain_frame,
        max_evals=max_evals,
        batch_size=256,
        silent=True,
    )
    values = np.asarray(explanation.values, dtype=float)
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    predictions = final_bundle.predict_tn(explain_frame)
    additivity_error = float(np.max(np.abs(base_values + values.sum(axis=1) - predictions)))
    if (
        values.shape != explain_frame.shape
        or additivity_error > tn_shap_additivity_tolerance
    ):
        raise FinalXAIError(
            f"TN permutation-SHAP failed shape/additivity validation: {values.shape}, "
            f"error={additivity_error:.3g}, "
            f"tolerance={tn_shap_additivity_tolerance:.3g}"
        )
    tn_records: list[dict[str, Any]] = []
    explained_dates = dates.iloc[explain_indices].reset_index(drop=True)
    for feature_index, feature in enumerate(feature_names):
        for row_index, date in enumerate(explained_dates):
            tn_records.append(
                {
                    "target": "TN_out",
                    "model": "Ensemble_Huber",
                    "Date": date,
                    "feature": feature,
                    "feature_value": float(explain_frame.iloc[row_index, feature_index]),
                    "shap_value": float(values[row_index, feature_index]),
                    "base_value": float(base_values[row_index]),
                    "prediction": float(predictions[row_index]),
                    "explanation_scope": "full_data_refit_representative_rows",
                    "background_rows": len(background),
                    "permutation_rounds": int(tn_shap_permutations),
                }
            )
    tn_shap = pd.DataFrame.from_records(tn_records)
    tn_global = (
        tn_shap.groupby("feature", observed=True)
        .agg(
            mean_abs_shap=("shap_value", lambda x: float(np.mean(np.abs(x)))),
            mean_signed_shap=("shap_value", "mean"),
            n_rows=("Date", "nunique"),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    tn_global["importance_share"] = tn_global["mean_abs_shap"] / tn_global[
        "mean_abs_shap"
    ].sum()
    dec_global = (
        crossfitted.dec_shap_values.groupby("feature", observed=True)
        .agg(
            mean_abs_shap=("shap_value", lambda x: float(np.mean(np.abs(x)))),
            mean_signed_shap=("shap_value", "mean"),
            n_explanations=("shap_value", "size"),
            n_outer_seeds=("seed", "nunique"),
        )
        .reset_index()
        .sort_values("mean_abs_shap", ascending=False)
    )
    dec_global["importance_share"] = dec_global["mean_abs_shap"] / dec_global[
        "mean_abs_shap"
    ].sum()

    ale_records: list[pd.DataFrame] = []
    for target in ("TN_out", "DEC"):
        bundle = feature_set.bundles[target]
        X = bundle.matrix(FEATURE_KEY).reset_index(drop=True)
        estimator = final_bundle.tn_model if target == "TN_out" else final_bundle.dec_model
        for feature in ale_features:
            if feature not in X:
                continue
            grid = fit_ale_grid(
                X,
                feature,
                quantile_bins=int(ale_quantile_bins),
                support_lower_quantile=float(ale_support_lower_quantile),
                support_upper_quantile=float(ale_support_upper_quantile),
            )
            result = compute_ale_1d(estimator, X, grid)
            curve = result.curve.copy()
            curve.insert(0, "target", target)
            curve["n_in_support"] = result.n_in_support
            curve["weighted_centering_error"] = result.weighted_centering_error
            curve["interpretation"] = "model_association_within_empirical_support_not_causal"
            ale_records.append(curve)
    parent_predictions = pd.read_csv(
        parent_run / "outer_predictions.csv", encoding="utf-8-sig", parse_dates=["Date"]
    )
    local_cases = _local_cases(parent_predictions)
    tn_audit = pd.DataFrame(
        [
            {
                "target": "TN_out",
                "model": "Ensemble_Huber",
                "explained_rows": len(explain_frame),
                "background_rows": len(background),
                "features": len(feature_names),
                "max_evals_per_row": max_evals,
                "max_additivity_error": additivity_error,
                "additivity_tolerance_mg_L": tn_shap_additivity_tolerance,
                "prediction_reproduction_error": 0.0,
                "passed": True,
                "scope": "full_data_refit_representative_rows_not_outer_performance",
            }
        ]
    )
    return DeploymentXAIResult(
        tn_shap_values=tn_shap,
        tn_shap_audit=tn_audit,
        tn_shap_global=tn_global,
        dec_shap_global=dec_global,
        ale_curves=pd.concat(ale_records, ignore_index=True),
        local_cases=local_cases,
    )
