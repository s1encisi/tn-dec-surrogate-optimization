"""Frozen full-data proxy refit for the final TN_out and DEC workflow.

The deployment candidates are fixed from the completed random-interpolation
experiment. Hyperparameters are never selected from outer-test scores here.
The fitted artifacts carry no future-forecasting or independent-test claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import HuberRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from .metrics import regression_metrics
from .paired_random_windows import BASE_MODELS, FEATURE_KEY, PairedFeatureSet


FINAL_WINDOW = "2023_2025"
FINAL_DEC_MODEL = "ExtraTrees"
FINAL_TN_MODEL = "Ensemble_Huber"

_NEUTRAL_TN_FEATURES = (
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
_NEUTRAL_DEC_FEATURES = tuple(
    feature for feature in _NEUTRAL_TN_FEATURES if feature != "TN_in"
)
_PRIOR_ARTIFACT_SCHEMAS = {
    "69582ea9d9f14ceb321b30dc556cc2e8614a3c4ddedb15b788117ea461cfe70a": (
        _NEUTRAL_TN_FEATURES
    ),
    "364ce4b63548343a0126489844a1f32e30edb3482d4e1986df85aaa3938568f1": (
        _NEUTRAL_DEC_FEATURES
    ),
}


class FinalProxyError(ValueError):
    """Raised when a frozen refit contract would be violated."""


def _seeded_clone(estimator: Any, seed: int) -> Any:
    fitted = clone(estimator)
    available = fitted.get_params(deep=True)
    seed_parameters = {
        key: int(seed)
        for key in available
        if key.endswith("random_state") or key.endswith("random_seed")
    }
    return fitted.set_params(**seed_parameters)


def _as_candidate_index(value: Any) -> int:
    index = int(float(value))
    if index < 0:
        raise FinalProxyError("Candidate indices must be non-negative.")
    return index


def freeze_candidate_table(
    tuning_trials: pd.DataFrame,
    *,
    target: str = "TN_out",
    training_window: str = FINAL_WINDOW,
    model_order: Sequence[str] = BASE_MODELS,
) -> pd.DataFrame:
    """Freeze one base-model candidate using mean inner-OOF errors only.

    For every candidate, pooled inner-OOF RMSE is averaged across the five
    outer-training splits. Mean inner-OOF MAE and then candidate index are
    deterministic tie-breakers. Outer-test predictions are never read.
    """

    required = {
        "target",
        "training_window",
        "seed",
        "model",
        "candidate_index",
        "record_type",
        "inner_RMSE",
        "inner_MAE",
        "status",
        "outer_test_accessed",
    }
    missing = required.difference(tuning_trials.columns)
    if missing:
        raise FinalProxyError(f"Tuning table lacks required columns: {sorted(missing)}")
    selected = tuning_trials.loc[
        tuning_trials["target"].eq(target)
        & tuning_trials["training_window"].eq(training_window)
        & tuning_trials["record_type"].eq("pooled_inner_oof")
        & tuning_trials["status"].eq("completed")
        & tuning_trials["model"].isin(tuple(model_order))
    ].copy()
    if selected.empty:
        raise FinalProxyError("No completed pooled inner-OOF tuning records were found.")
    accessed = selected["outer_test_accessed"].astype(str).str.lower().isin({"true", "1"})
    if accessed.any():
        raise FinalProxyError("A candidate-selection row accessed the outer test.")
    selected["candidate_index"] = selected["candidate_index"].map(_as_candidate_index)
    selected["inner_RMSE"] = pd.to_numeric(selected["inner_RMSE"], errors="raise")
    selected["inner_MAE"] = pd.to_numeric(selected["inner_MAE"], errors="raise")
    summary = (
        selected.groupby(["model", "candidate_index"], observed=True)
        .agg(
            mean_inner_oof_RMSE=("inner_RMSE", "mean"),
            mean_inner_oof_MAE=("inner_MAE", "mean"),
            sd_inner_oof_RMSE=("inner_RMSE", "std"),
            n_outer_train_splits=("seed", "nunique"),
        )
        .reset_index()
    )
    rows: list[pd.Series] = []
    for model in model_order:
        model_rows = summary.loc[summary["model"].eq(model)].sort_values(
            ["mean_inner_oof_RMSE", "mean_inner_oof_MAE", "candidate_index"],
            kind="stable",
        )
        if model_rows.empty:
            raise FinalProxyError(f"Missing candidate records for {model}.")
        row = model_rows.iloc[0].copy()
        row["selection_rule"] = (
            "minimum_mean_pooled_inner_oof_RMSE_then_mean_MAE_then_candidate_index"
        )
        row["outer_test_accessed_for_locking"] = False
        rows.append(row)
    frozen = pd.DataFrame(rows).reset_index(drop=True)
    if tuple(frozen["model"]) != tuple(model_order):
        raise RuntimeError("Frozen candidate order differs from the registered model order.")
    return frozen


def align_feature_frame(X: Any, feature_names: Sequence[str]) -> pd.DataFrame:
    """Select a frozen contract and explicitly adapt the prior artifact schema."""

    expected = tuple(str(value) for value in feature_names)
    if not isinstance(X, pd.DataFrame):
        return pd.DataFrame(X, columns=expected)
    if set(expected).issubset(X.columns):
        return X.loc[:, list(expected)]
    candidate = (
        X.drop(columns=["TN_in"])
        if "TN_in" in X and len(X.columns) == len(expected) + 1
        else X
    )
    observed = tuple(str(value) for value in candidate.columns)
    expected_digest = sha256("\0".join(expected).encode()).hexdigest()
    if _PRIOR_ARTIFACT_SCHEMAS.get(expected_digest) == observed:
        aligned = candidate.copy()
        aligned.columns = expected
        return aligned
    missing = sorted(set(expected).difference(candidate.columns))
    raise FinalProxyError(f"Proxy input lacks features from its frozen contract: {missing}")


@dataclass
class FinalHuberEnsemble:
    """Full-data base models plus an OOF-fitted robust linear meta learner."""

    feature_names: tuple[str, ...]
    component_order: tuple[str, ...]
    base_models: Mapping[str, Any]
    meta_scaler: StandardScaler
    meta_model: HuberRegressor

    def _validated_frame(self, X: pd.DataFrame) -> pd.DataFrame:
        return align_feature_frame(X, self.feature_names)

    def component_predictions(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = self._validated_frame(X)
        values = {
            name: np.asarray(self.base_models[name].predict(frame), dtype=float).reshape(-1)
            for name in self.component_order
        }
        predictions = pd.DataFrame(values, index=frame.index)
        if not np.isfinite(predictions.to_numpy(float)).all():
            raise FinalProxyError("A TN ensemble component returned an invalid prediction.")
        return predictions

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        components = self.component_predictions(X)
        prediction = np.asarray(
            self.meta_model.predict(self.meta_scaler.transform(components)), dtype=float
        ).reshape(-1)
        if not np.isfinite(prediction).all():
            raise FinalProxyError("TN Ensemble_Huber returned invalid predictions.")
        return prediction

    @property
    def effective_weights(self) -> np.ndarray:
        return np.asarray(self.meta_model.coef_, dtype=float) / np.asarray(
            self.meta_scaler.scale_, dtype=float
        )

    @property
    def effective_intercept(self) -> float:
        weights = self.effective_weights
        return float(self.meta_model.intercept_ - np.dot(weights, self.meta_scaler.mean_))


@dataclass
class FinalProxyBundle:
    """The two selected fitted proxies and their frozen feature contract."""

    dec_model: Any
    tn_model: FinalHuberEnsemble
    dec_feature_names: tuple[str, ...]
    tn_feature_names: tuple[str, ...]
    deployment_seed: int
    training_dates: tuple[str, ...]

    def predict_dec(self, X: pd.DataFrame) -> np.ndarray:
        frame = align_feature_frame(X, self.dec_feature_names)
        return np.asarray(self.dec_model.predict(frame), dtype=float).reshape(-1)

    def predict_tn(self, X: pd.DataFrame) -> np.ndarray:
        return self.tn_model.predict(align_feature_frame(X, self.tn_feature_names))


@dataclass(frozen=True)
class FinalProxyFitResult:
    bundle: FinalProxyBundle
    candidate_lock: pd.DataFrame
    oof_predictions: pd.DataFrame
    oof_reconstruction_metrics: pd.DataFrame
    ensemble_weights: pd.DataFrame
    fold_assignments: pd.DataFrame
    full_fit_predictions: pd.DataFrame


def _candidate_parameters(
    registry: Mapping[str, Any], candidate_lock: pd.DataFrame
) -> dict[str, dict[str, Any]]:
    parameters: dict[str, dict[str, Any]] = {}
    for row in candidate_lock.itertuples(index=False):
        model = str(row.model)
        if model not in registry:
            raise FinalProxyError(f"Frozen model {model} is absent from the registry.")
        index = _as_candidate_index(row.candidate_index)
        candidates = registry[model].candidates
        if index >= len(candidates):
            raise FinalProxyError(f"Frozen candidate {index} is invalid for {model}.")
        parameters[model] = dict(candidates[index])
    return parameters


def fit_final_proxy_bundle(
    feature_set: PairedFeatureSet,
    registries: Mapping[str, Mapping[str, Any]],
    candidate_lock: pd.DataFrame,
    *,
    deployment_seed: int = 20260823,
    inner_folds: int = 3,
    dec_candidate_index: int = 0,
) -> FinalProxyFitResult:
    """Refit the selected proxies on all 1,076 eligible three-year rows."""

    if inner_folds != 3:
        raise FinalProxyError("The final Huber reconstruction requires three OOF folds.")
    if tuple(registries) != ("TN_out", "DEC"):
        raise FinalProxyError("Registries must be ordered as TN_out then DEC.")
    tn_bundle = feature_set.bundles["TN_out"]
    dec_bundle = feature_set.bundles["DEC"]
    dates = pd.DatetimeIndex(pd.to_datetime(tn_bundle.anchor["Date"])).normalize()
    if not dates.equals(pd.DatetimeIndex(pd.to_datetime(dec_bundle.anchor["Date"]))):
        raise FinalProxyError("TN and DEC final refits must share one date panel.")
    all_indices = np.arange(len(dates), dtype=int)
    if len(all_indices) != 1076:
        raise FinalProxyError(
            f"Expected 1,076 common initial-data rows, observed {len(all_indices)}."
        )

    tn_registry = registries["TN_out"]
    dec_registry = registries["DEC"]
    if tuple(name for name in tn_registry if name in BASE_MODELS) != tuple(BASE_MODELS):
        raise FinalProxyError("TN registry does not contain the frozen 15-model order.")
    parameter_lock = _candidate_parameters(tn_registry, candidate_lock)
    X_tn = tn_bundle.matrix(FEATURE_KEY)
    y_tn = tn_bundle.anchor["actual"].to_numpy(float)
    splitter = KFold(
        n_splits=inner_folds,
        shuffle=True,
        random_state=int(deployment_seed) + 300_003,
    )
    folds = list(splitter.split(all_indices))
    base_oof = pd.DataFrame(index=np.arange(len(all_indices)), columns=BASE_MODELS, dtype=float)
    fitted_base: dict[str, Any] = {}
    fold_records: list[dict[str, Any]] = []
    for fold_id, (_, validation_position) in enumerate(folds, start=1):
        validation_set = set(int(value) for value in validation_position)
        for index in all_indices:
            fold_records.append(
                {
                    "inner_fold": fold_id,
                    "row_index": int(index),
                    "Date": dates[index],
                    "role": "inner_validation" if int(index) in validation_set else "inner_train",
                }
            )
    for model in BASE_MODELS:
        spec = tn_registry[model]
        parameters = parameter_lock[model]
        for fit_position, validation_position in folds:
            estimator = _seeded_clone(spec.estimator, deployment_seed).set_params(**parameters)
            estimator.fit(X_tn.iloc[fit_position], y_tn[fit_position])
            prediction = np.asarray(
                estimator.predict(X_tn.iloc[validation_position]), dtype=float
            ).reshape(-1)
            if len(prediction) != len(validation_position) or not np.isfinite(prediction).all():
                raise FinalProxyError(f"Invalid full-data OOF predictions from {model}.")
            base_oof.loc[validation_position, model] = prediction
        full_estimator = _seeded_clone(spec.estimator, deployment_seed).set_params(**parameters)
        full_estimator.fit(X_tn, y_tn)
        fitted_base[model] = full_estimator
    if base_oof.isna().any().any():
        raise FinalProxyError("The final Huber OOF matrix is incomplete.")

    meta_scaler = StandardScaler().fit(base_oof)
    meta_model = HuberRegressor(
        epsilon=1.35,
        alpha=0.001,
        max_iter=2_000,
        tol=1e-7,
    ).fit(meta_scaler.transform(base_oof), y_tn)
    tn_model = FinalHuberEnsemble(
        feature_names=tuple(tn_bundle.feature_names[FEATURE_KEY]),
        component_order=tuple(BASE_MODELS),
        base_models=fitted_base,
        meta_scaler=meta_scaler,
        meta_model=meta_model,
    )
    tn_oof_prediction = np.asarray(
        meta_model.predict(meta_scaler.transform(base_oof)), dtype=float
    ).reshape(-1)

    dec_spec = dec_registry[FINAL_DEC_MODEL]
    if dec_candidate_index < 0 or dec_candidate_index >= len(dec_spec.candidates):
        raise FinalProxyError("The frozen DEC candidate index is invalid.")
    dec_parameters = dict(dec_spec.candidates[int(dec_candidate_index)])
    X_dec = dec_bundle.matrix(FEATURE_KEY)
    y_dec = dec_bundle.anchor["actual"].to_numpy(float)
    dec_model = _seeded_clone(dec_spec.estimator, deployment_seed).set_params(**dec_parameters)
    dec_model.fit(X_dec, y_dec)

    final_bundle = FinalProxyBundle(
        dec_model=dec_model,
        tn_model=tn_model,
        dec_feature_names=tuple(dec_bundle.feature_names[FEATURE_KEY]),
        tn_feature_names=tuple(tn_bundle.feature_names[FEATURE_KEY]),
        deployment_seed=int(deployment_seed),
        training_dates=tuple(dates.strftime("%Y-%m-%d")),
    )
    oof_frame = pd.DataFrame(
        {
            "Date": dates,
            "target": "TN_out",
            "actual": y_tn,
            "prediction": tn_oof_prediction,
            "role": "full_data_crossfit_meta_reconstruction_not_performance_estimate",
        }
    )
    oof_metrics = pd.DataFrame(
        [
            {
                "target": "TN_out",
                **regression_metrics(y_tn, tn_oof_prediction),
                "metric_role": "meta_reconstruction_only_not_outer_or_future_performance",
            }
        ]
    )
    full_predictions = pd.concat(
        [
            pd.DataFrame(
                {
                    "Date": dates,
                    "target": "TN_out",
                    "actual": y_tn,
                    "prediction": final_bundle.predict_tn(X_tn),
                }
            ),
            pd.DataFrame(
                {
                    "Date": dates,
                    "target": "DEC",
                    "actual": y_dec,
                    "prediction": final_bundle.predict_dec(X_dec),
                }
            ),
        ],
        ignore_index=True,
    )
    full_predictions["role"] = "in_sample_full_refit_not_performance_estimate"
    weights = pd.DataFrame(
        {
            "ensemble_model": FINAL_TN_MODEL,
            "component_model": BASE_MODELS,
            "effective_weight": tn_model.effective_weights,
            "effective_intercept": tn_model.effective_intercept,
            "meta_epsilon": 1.35,
            "meta_alpha": 0.001,
        }
    )
    return FinalProxyFitResult(
        bundle=final_bundle,
        candidate_lock=candidate_lock.copy(),
        oof_predictions=oof_frame,
        oof_reconstruction_metrics=oof_metrics,
        ensemble_weights=weights,
        fold_assignments=pd.DataFrame.from_records(fold_records),
        full_fit_predictions=full_predictions,
    )


def artifact_contract(bundle: FinalProxyBundle) -> dict[str, Any]:
    """Return a JSON-serializable inference contract for the fitted proxies."""

    date_payload = "\n".join(bundle.training_dates).encode("utf-8")
    return {
        "targets": {
            "DEC": {
                "model": FINAL_DEC_MODEL,
                "features": list(bundle.dec_feature_names),
                "unit": "kWh/d (project convention; meter boundary unresolved)",
            },
            "TN_out": {
                "model": FINAL_TN_MODEL,
                "features": list(bundle.tn_feature_names),
                "components": list(bundle.tn_model.component_order),
                "unit": "mg/L",
            },
        },
        "feature_version": FEATURE_KEY,
        "training_window": FINAL_WINDOW,
        "training_rows": len(bundle.training_dates),
        "training_date_sha256": sha256(date_payload).hexdigest(),
        "deployment_seed": bundle.deployment_seed,
        "prediction_method_evidence": "random_ordinary_daily_80_20_five_overlapping_seeds",
        "independent_test": False,
        "future_prediction_claim_authorized": False,
        "optimization_role": "historical_support_proxy_simulation_only",
    }
