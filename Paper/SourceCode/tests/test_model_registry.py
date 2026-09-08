from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from taici.model_registry import (
    CLASSICAL_MODEL_NAMES,
    SCALED_MODEL_NAMES,
    TABULAR_MODEL_NAMES,
    LegalLagAutoRegressor,
    PersistenceRegressor,
    SeasonalNaiveRegressor,
    build_classical_model_registry,
    rolling_sarimax_one_step,
    sample_hyperparameter_candidates,
)


def test_registry_contains_exact_v3_classical_pool_and_fold_local_pipelines() -> None:
    registry = build_classical_model_registry(
        seed=71, candidate_count=5, n_features=8, target="TN_out"
    )

    assert tuple(registry) == CLASSICAL_MODEL_NAMES
    assert registry["SARIMAX"].estimator is None
    assert registry["SARIMAX"].requires_sequential_update
    for name in TABULAR_MODEL_NAMES:
        estimator = registry[name].estimator
        assert isinstance(estimator, Pipeline)
        assert isinstance(estimator.named_steps["imputer"], SimpleImputer)
        assert estimator.named_steps["imputer"].strategy == "median"
        assert estimator.named_steps["imputer"].add_indicator
        assert ("scaler" in estimator.named_steps) == (name in SCALED_MODEL_NAMES)
        if name in SCALED_MODEL_NAMES:
            assert isinstance(estimator.named_steps["scaler"], StandardScaler)


@pytest.mark.parametrize(
    "name",
    ["ElasticNet", "PLS", "SVR", "RandomForest", "ExtraTrees", "XGBoost", "LightGBM", "CatBoost", "SARIMAX"],
)
def test_random_candidates_are_reproducible_unique_and_settable(name: str) -> None:
    first = sample_hyperparameter_candidates(name, candidate_count=7, seed=101, n_features=6)
    second = sample_hyperparameter_candidates(name, candidate_count=7, seed=101, n_features=6)
    other_seed = sample_hyperparameter_candidates(name, candidate_count=7, seed=102, n_features=6)

    assert first == second
    assert len(first) == 7
    assert len({repr(sorted(candidate.items())) for candidate in first}) == 7
    assert first != other_seed
    if name != "SARIMAX":
        estimator = build_classical_model_registry(
            seed=101, candidate_count=7, n_features=6
        )[name].estimator
        for candidate in first:
            clone(estimator).set_params(**candidate)


def test_parameter_free_baselines_have_one_candidate_and_rf_alias_is_supported() -> None:
    assert sample_hyperparameter_candidates("Persistence", candidate_count=12) == ({},)
    assert sample_hyperparameter_candidates("SeasonalNaive", candidate_count=12) == ({},)
    assert sample_hyperparameter_candidates("RF", candidate_count=3) == (
        sample_hyperparameter_candidates("RandomForest", candidate_count=3)
    )


def test_persistence_and_seasonal_naive_use_only_the_registered_target_lag() -> None:
    X = pd.DataFrame(
        {
            "lag1_TN_out": [1.0, 2.0, 3.0],
            "lag7_TN_out": [10.0, 20.0, 30.0],
            "lag1_DEC": [999.0, 999.0, 999.0],
            "source_DO": [5.0, 6.0, 7.0],
        }
    )
    y = np.array([2.0, 3.0, 4.0])

    persistence = PersistenceRegressor(target="TN_out").fit(X, y)
    seasonal = SeasonalNaiveRegressor(target="TN_out").fit(X, y)

    np.testing.assert_array_equal(persistence.predict(X), X["lag1_TN_out"])
    np.testing.assert_array_equal(seasonal.predict(X), X["lag7_TN_out"])


def test_lag_baseline_rejects_ambiguous_targets_and_missing_observed_lag() -> None:
    ambiguous = pd.DataFrame({"lag1_TN_out": [1.0], "lag1_DEC": [2.0]})
    with pytest.raises(ValueError, match="ambiguous"):
        PersistenceRegressor().fit(ambiguous, np.array([1.0]))

    frame = pd.DataFrame({"lag1_TN_out": [1.0, np.nan]})
    model = PersistenceRegressor(target="TN_out").fit(frame, np.array([1.0, 2.0]))
    with pytest.raises(ValueError, match="cannot be imputed"):
        model.predict(frame)


def test_autoreg_fits_only_legal_lags_and_ignores_exogenous_columns() -> None:
    rng = np.random.default_rng(4)
    lag1 = rng.normal(size=100)
    lag7 = rng.normal(size=100)
    frame = pd.DataFrame(
        {
            "source_DO": rng.normal(size=100) * 1_000,
            "lag1_TN_out": lag1,
            "lag7_TN_out": lag7,
            "same_day_TN_out": rng.normal(size=100) * 1_000,
        }
    )
    y = 2.0 * lag1 - 0.5 * lag7 + 3.0
    model = LegalLagAutoRegressor(lag_days=(1, 7), target="TN_out").fit(frame, y)

    altered = frame.copy()
    altered["source_DO"] *= -100
    altered["same_day_TN_out"] += 1e9
    np.testing.assert_allclose(model.predict(altered), y, atol=1e-10)
    assert model.lag_columns_ == ("lag1_TN_out", "lag7_TN_out")


def test_autoreg_and_baselines_reject_nonfinite_training_labels() -> None:
    frame = pd.DataFrame(
        {"lag1_TN_out": [1.0, 2.0], "lag7_TN_out": [1.0, 2.0]}
    )
    y = np.array([1.0, np.nan])
    with pytest.raises(ValueError, match="target imputation is forbidden"):
        PersistenceRegressor(target="TN_out").fit(frame, y)
    with pytest.raises(ValueError, match="target imputation is forbidden"):
        LegalLagAutoRegressor(lag_days=(1, 7), target="TN_out").fit(frame, y)


def test_scaled_pipeline_fits_missing_features_without_mutating_input() -> None:
    X = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0],
            "b": [2.0, np.nan, 1.0, 0.0, 2.0, 3.0],
        }
    )
    original = X.copy(deep=True)
    y = np.array([1.0, 1.5, 1.2, 2.0, 2.3, 2.8])
    model = build_classical_model_registry(
        seed=8, candidate_count=2, n_features=2
    )["ElasticNet"].estimator

    predictions = model.fit(X.iloc[:5], y[:5]).predict(X.iloc[5:])

    assert np.isfinite(predictions).all()
    pd.testing.assert_frame_equal(X, original)
    assert model.named_steps["imputer"].statistics_.shape == (2,)


def test_sarimax_forecasts_before_observing_each_evaluation_target() -> None:
    train = np.array([0.8**index for index in range(35)], dtype=float)
    future_a = np.array([0.01, 0.02, 0.03, 0.04])
    future_b = future_a.copy()
    future_b[0] = 100.0

    predictions_a = rolling_sarimax_one_step(
        train, future_a, order=(1, 0, 0), trend="n", maxiter=100
    )
    predictions_b = rolling_sarimax_one_step(
        train, future_b, order=(1, 0, 0), trend="n", maxiter=100
    )

    assert predictions_a.shape == future_a.shape
    assert np.isfinite(predictions_a).all()
    assert predictions_a[0] == pytest.approx(predictions_b[0])
    assert predictions_a[1] != pytest.approx(predictions_b[1])


def test_sarimax_exogenous_preprocessing_does_not_fit_on_evaluation_rows() -> None:
    train_y = np.linspace(1.0, 3.0, 30)
    train_y[8] = np.nan
    future_y = np.linspace(3.1, 3.4, 4)
    train_x = np.column_stack([np.linspace(0.0, 1.0, 30), np.full(30, np.nan)])
    future_x_a = np.array([[1.1, np.nan], [1.2, np.nan], [1.3, np.nan], [1.4, np.nan]])
    future_x_b = future_x_a.copy()
    future_x_b[1:, 0] = 1e9

    predictions_a = rolling_sarimax_one_step(
        train_y,
        future_y,
        train_x,
        future_x_a,
        order=(1, 0, 0),
        trend="c",
        maxiter=100,
    )
    predictions_b = rolling_sarimax_one_step(
        train_y,
        future_y,
        train_x,
        future_x_b,
        order=(1, 0, 0),
        trend="c",
        maxiter=100,
    )

    assert predictions_a[0] == pytest.approx(predictions_b[0])


@pytest.mark.parametrize("candidate_count", [0, -1])
def test_candidate_count_must_be_positive(candidate_count: int) -> None:
    with pytest.raises(ValueError, match="at least one"):
        sample_hyperparameter_candidates("SVR", candidate_count=candidate_count)
