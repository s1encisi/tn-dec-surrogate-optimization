from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from taici.fast_proxy import FastExactProxyBundle, audit_fast_proxy_fidelity
from taici.final_proxy import (
    FinalHuberEnsemble,
    FinalProxyBundle,
    FinalProxyError,
    freeze_candidate_table,
)


class _ColumnRegressor:
    def __init__(self, column: str, multiplier: float = 1.0) -> None:
        self.column = column
        self.multiplier = float(multiplier)

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[self.column].to_numpy(float) * self.multiplier


class _LinearMetaModel:
    def __init__(self, coefficients: list[float], intercept: float) -> None:
        self.coef_ = np.asarray(coefficients, dtype=float)
        self.intercept_ = float(intercept)

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(matrix, dtype=float) @ self.coef_ + self.intercept_


def _tuning_rows() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    errors = {
        "ModelA": {0: (2.0, 1.2), 1: (1.0, 0.8)},
        "ModelB": {0: (3.0, 1.0), 1: (3.0, 1.1)},
    }
    for seed in (11, 37):
        for model, candidates in errors.items():
            for candidate_index, (rmse, mae) in candidates.items():
                records.append(
                    {
                        "target": "TN_out",
                        "training_window": "2023_2025",
                        "seed": seed,
                        "model": model,
                        "candidate_index": candidate_index,
                        "record_type": "pooled_inner_oof",
                        "inner_RMSE": rmse,
                        "inner_MAE": mae,
                        "status": "completed",
                        "outer_test_accessed": False,
                    }
                )
    return pd.DataFrame.from_records(records)


def test_freeze_candidate_table_uses_inner_oof_and_deterministic_ties() -> None:
    trials = _tuning_rows()
    # An outer-test row must not influence the lock, even with an unrealistically low error.
    trials.loc[len(trials)] = {
        "target": "TN_out",
        "training_window": "2023_2025",
        "seed": 11,
        "model": "ModelA",
        "candidate_index": 0,
        "record_type": "outer_test",
        "inner_RMSE": -100.0,
        "inner_MAE": -100.0,
        "status": "completed",
        "outer_test_accessed": True,
    }

    frozen = freeze_candidate_table(trials, model_order=("ModelA", "ModelB"))

    assert frozen["model"].tolist() == ["ModelA", "ModelB"]
    assert frozen["candidate_index"].astype(int).tolist() == [1, 0]
    assert not frozen["outer_test_accessed_for_locking"].any()


def test_freeze_candidate_table_rejects_inner_record_that_accessed_outer_test() -> None:
    trials = _tuning_rows()
    trials.loc[0, "outer_test_accessed"] = True

    with pytest.raises(FinalProxyError, match="accessed the outer test"):
        freeze_candidate_table(trials, model_order=("ModelA", "ModelB"))


def test_final_huber_ensemble_preserves_component_and_feature_contracts() -> None:
    frame = pd.DataFrame({"f1": [1.0, 3.0], "f2": [2.0, 4.0], "unused": [9.0, 9.0]})
    component_frame = pd.DataFrame(
        {"first": frame["f1"], "second": 2.0 * frame["f2"]},
        index=frame.index,
    )
    scaler = StandardScaler().fit(component_frame)
    meta = _LinearMetaModel([0.75, -0.25], intercept=1.5)
    ensemble = FinalHuberEnsemble(
        feature_names=("f1", "f2"),
        component_order=("first", "second"),
        base_models={
            "first": _ColumnRegressor("f1"),
            "second": _ColumnRegressor("f2", multiplier=2.0),
        },
        meta_scaler=scaler,
        meta_model=meta,  # type: ignore[arg-type]
    )

    expected = meta.predict(scaler.transform(component_frame))
    assert np.allclose(ensemble.predict(frame), expected)
    assert ensemble.component_predictions(frame).columns.tolist() == ["first", "second"]
    assert np.allclose(ensemble.effective_weights, meta.coef_ / scaler.scale_)

    with pytest.raises(FinalProxyError, match="lacks features"):
        ensemble.predict(frame.drop(columns="f2"))


def test_fast_proxy_preserves_fitted_tree_ensembles_and_adaboost() -> None:
    rng = np.random.default_rng(17)
    frame = pd.DataFrame(rng.normal(size=(40, 2)), columns=["f1", "f2"])
    target = frame["f1"].to_numpy() - 0.5 * frame["f2"].to_numpy()

    def fitted(model: object) -> Pipeline:
        pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("model", model),
            ]
        )
        return pipeline.fit(frame, target)

    components = {
        "RandomForest": fitted(
            RandomForestRegressor(n_estimators=9, random_state=1, min_samples_leaf=2)
        ),
        "ExtraTrees": fitted(
            ExtraTreesRegressor(n_estimators=9, random_state=2, min_samples_leaf=2)
        ),
        "AdaBoost": fitted(AdaBoostRegressor(n_estimators=9, random_state=3)),
    }
    component_frame = pd.DataFrame(
        {name: model.predict(frame) for name, model in components.items()}
    )
    scaler = StandardScaler().fit(component_frame)
    meta = _LinearMetaModel([0.2, 0.7, 0.1], intercept=0.4)
    tn_model = FinalHuberEnsemble(
        feature_names=("f1", "f2"),
        component_order=tuple(components),
        base_models=components,
        meta_scaler=scaler,
        meta_model=meta,  # type: ignore[arg-type]
    )
    original = FinalProxyBundle(
        dec_model=components["ExtraTrees"],
        tn_model=tn_model,
        dec_feature_names=("f1", "f2"),
        tn_feature_names=("f1", "f2"),
        deployment_seed=17,
        training_dates=tuple(pd.date_range("2025-01-01", periods=len(frame)).astype(str)),
    )
    fast = FastExactProxyBundle.from_frozen(original)

    audit = audit_fast_proxy_fidelity(original, fast, frame, frame, tolerance=1e-10)

    assert audit["status"] == "PASSED"
    assert np.allclose(fast.predict_tn(frame), original.predict_tn(frame), atol=1e-12)
    assert np.allclose(fast.predict_dec(frame), original.predict_dec(frame), atol=1e-12)
