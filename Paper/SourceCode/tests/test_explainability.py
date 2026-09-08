from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from taici.explainability import (
    ExplainabilityInputError,
    aggregate_grouped_shap,
    block_bootstrap_ale_1d,
    block_permutation_indices,
    compute_ale_1d,
    compute_tree_shap,
    fit_ale_grid,
    joint_group_block_permutation,
    pipeline_feature_mapping,
    select_preregistered_local_cases,
    summarize_fold_stability,
    transform_pipeline_design,
    validate_development_only,
    validate_phase7_scope,
)


FEATURES = ("source_TN", "source_DO", "lag1_TN_out")
GROUPS = {
    "influent_load": ("source_TN",),
    "operation_proxy": ("source_DO",),
    "target_history": ("lag1_TN_out",),
}


def development_fixture(n_rows: int = 84) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")
    time = np.arange(n_rows, dtype=float)
    frame = pd.DataFrame(
        {
            "source_TN": 25.0 + 4.0 * np.sin(time / 8.0),
            "source_DO": 2.0 + 0.3 * np.cos(time / 6.0),
            "lag1_TN_out": 9.0 + 0.04 * time,
        }
    )
    frame.loc[[3, 20], "source_DO"] = np.nan
    target = (
        0.45 * frame["source_TN"].to_numpy()
        + 1.8 * frame["lag1_TN_out"].to_numpy()
        + np.nan_to_num(frame["source_DO"].to_numpy(), nan=2.0)
    )
    return frame, target, pd.Series(dates)


def fitted_pipeline() -> tuple[Pipeline, pd.DataFrame, np.ndarray, pd.Series]:
    frame, target, dates = development_fixture()
    pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median", add_indicator=True, keep_empty_features=True
                ),
            ),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=80,
                    min_samples_leaf=2,
                    random_state=7,
                    n_jobs=1,
                ),
            ),
        ]
    )
    pipeline.fit(frame.loc[:, list(FEATURES)], target)
    return pipeline, frame, target, dates


def test_development_scope_rejects_fixed_test_and_wrong_target() -> None:
    frame = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2025-06-29", "2025-06-30"]),
            "study_partition": ["development", "development"],
            "target": ["TN_out", "TN_out"],
        }
    )
    validated = validate_development_only(
        frame,
        development_start="2023-01-01",
        development_end="2025-06-30",
        target_column="target",
    )
    assert validated["Date"].max() == pd.Timestamp("2025-06-30")

    fixed = frame.copy()
    fixed.loc[1, "study_partition"] = "fixed_test"
    with pytest.raises(ExplainabilityInputError, match="development rows only"):
        validate_development_only(
            fixed,
            development_start="2023-01-01",
            development_end="2025-06-30",
        )

    wrong_target = frame.copy()
    wrong_target.loc[1, "target"] = "DEC"
    with pytest.raises(ExplainabilityInputError, match="authorized for TN_out only"):
        validate_development_only(
            wrong_target,
            development_start="2023-01-01",
            development_end="2025-06-30",
            target_column="target",
        )

    with pytest.raises(ExplainabilityInputError, match="primary explanation model"):
        validate_phase7_scope(
            frame,
            target="TN_out",
            model_name="RandomForest",
            development_start="2023-01-01",
            development_end="2025-06-30",
        )


def test_pipeline_mapping_gives_missing_indicator_its_source_group() -> None:
    pipeline, frame, _, _ = fitted_pipeline()

    mapping = pipeline_feature_mapping(pipeline, FEATURES, GROUPS)
    design = transform_pipeline_design(pipeline, frame, FEATURES, GROUPS)

    indicator = mapping.loc[mapping["is_missing_indicator"]]
    assert len(indicator) == 1
    assert indicator.iloc[0]["source_feature"] == "source_DO"
    assert indicator.iloc[0]["physical_group"] == "operation_proxy"
    assert design.matrix.shape == (len(frame), len(mapping))
    assert np.isfinite(design.matrix).all()


def test_treeshap_additivity_and_grouped_contribution_conservation() -> None:
    pipeline, frame, _, _ = fitted_pipeline()

    result = compute_tree_shap(
        pipeline,
        frame.iloc[:25],
        FEATURES,
        GROUPS,
        additivity_tolerance=1e-8,
    )
    grouped = aggregate_grouped_shap(
        result.values, result.transformed_design.metadata
    )

    assert result.max_additivity_error <= 1e-8
    np.testing.assert_allclose(result.predictions, result.reconstructed_predictions)
    np.testing.assert_allclose(
        grouped.local_values.sum(axis=1).to_numpy(), result.values.sum(axis=1)
    )
    assert grouped.max_conservation_error <= 1e-10
    assert set(grouped.global_importance["physical_group"]) == set(GROUPS)


def test_block_permutation_preserves_within_block_offsets_and_terminal_remainder() -> None:
    indices = block_permutation_indices(30, 7, np.random.default_rng(11))

    assert sorted(indices.tolist()) == list(range(30))
    assert indices[-2:].tolist() == [28, 29]
    for block in indices[:28].reshape(4, 7):
        assert np.diff(block).tolist() == [1] * 6


def test_joint_group_block_permutation_is_development_only_and_reproducible() -> None:
    pipeline, frame, target, dates = fitted_pipeline()
    kwargs = {
        "development_start": "2023-01-01",
        "development_end": "2025-06-30",
        "block_length": 7,
        "repeats": 8,
        "random_seed": 17,
    }

    first = joint_group_block_permutation(
        pipeline, frame, target, dates, GROUPS, **kwargs
    )
    second = joint_group_block_permutation(
        pipeline, frame, target, dates, GROUPS, **kwargs
    )

    pd.testing.assert_frame_equal(first.repeats, second.repeats)
    assert len(first.repeats) == len(GROUPS) * 8
    assert first.repeats["fixed_test_accessed"].eq(False).all()
    assert first.summary["mean_delta_RMSE"].max() > 0

    partitions = pd.Series("development", index=frame.index)
    partitions.iloc[-1] = "fixed_test"
    with pytest.raises(ExplainabilityInputError, match="development rows only"):
        joint_group_block_permutation(
            pipeline,
            frame,
            target,
            dates,
            GROUPS,
            study_partition=partitions,
            **kwargs,
        )


def test_ale_uses_frozen_training_edges_and_is_weight_centered() -> None:
    pipeline, frame, _, _ = fitted_pipeline()
    grid = fit_ale_grid(
        frame,
        "source_TN",
        quantile_bins=6,
        support_lower_quantile=0.05,
        support_upper_quantile=0.95,
    )
    original_edges = grid.edges
    result = compute_ale_1d(pipeline, frame, grid)

    changed_evaluation = frame.copy()
    changed_evaluation.loc[0, "source_TN"] = 1_000_000.0
    changed = compute_ale_1d(pipeline, changed_evaluation, grid)

    assert grid.edges == original_edges
    assert result.grid is grid
    assert changed.grid.edges == original_edges
    assert result.weighted_centering_error < 1e-12
    assert result.curve["n"].sum() == result.n_in_support
    assert result.curve["within_training_support"].all()


def test_ale_block_bootstrap_reuses_frozen_grid_and_never_reads_test() -> None:
    pipeline, frame, _, dates = fitted_pipeline()
    grid = fit_ale_grid(frame, "source_TN", quantile_bins=5)

    result = block_bootstrap_ale_1d(
        pipeline,
        frame,
        grid,
        dates,
        development_start="2023-01-01",
        development_end="2025-06-30",
        replicates=6,
        block_length=7,
        random_seed=31,
    )

    assert result.point.grid is grid
    assert result.samples["replicate"].nunique() == 6
    assert result.samples["fixed_test_accessed"].eq(False).all()
    assert result.interval["replicates"].eq(6).all()
    assert len(result.interval) == grid.effective_bins


def test_fold_stability_uses_fixed_top_k_and_pairwise_spearman() -> None:
    records = []
    values = {
        "F1": {"a": (3.0, 1.0), "b": (2.0, -1.0), "c": (1.0, 0.2)},
        "F2": {"a": (2.8, 0.8), "b": (1.0, -0.7), "c": (2.0, 0.1)},
        "F3": {"a": (3.2, 1.2), "b": (2.1, -0.6), "c": (0.8, -0.1)},
    }
    for fold, features in values.items():
        for feature, (absolute, signed) in features.items():
            records.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "mean_abs_shap": absolute,
                    "mean_signed_shap": signed,
                }
            )

    result = summarize_fold_stability(pd.DataFrame(records), top_k=2)
    summary = result.feature_summary.set_index("feature")

    assert summary.loc["a", "top_k_frequency"] == 1.0
    assert summary.loc["a", "dominant_direction"] == "positive"
    assert summary.loc["b", "dominant_direction"] == "negative"
    assert len(result.pairwise_spearman) == 3
    assert -1 <= result.median_pairwise_spearman <= 1


def test_local_cases_follow_fixed_rules_and_earliest_tie_break() -> None:
    dates = pd.date_range("2024-02-01", periods=10, freq="D")
    actual = np.arange(1.0, 11.0)
    prediction = actual + np.array([0.1, 0.2, 0.3, 0.4, 0.01, 0.01, 0.7, 0.8, 0.9, 4.0])
    frame = pd.DataFrame(
        {
            "Date": dates,
            "actual": actual,
            "prediction": prediction,
            "study_partition": "development",
        }
    )

    selected = select_preregistered_local_cases(
        frame,
        development_start="2023-01-01",
        development_end="2025-06-30",
    ).set_index("case_type")

    assert selected.loc["typical_accurate", "Date"] == dates[4]
    assert selected.loc["highest_target", "Date"] == dates[9]
    assert selected.loc["largest_error", "Date"] == dates[9]
    assert selected["fixed_test_accessed"].eq(False).all()

    fixed = frame.copy()
    fixed.loc[0, "study_partition"] = "fixed_test"
    with pytest.raises(ExplainabilityInputError, match="development rows only"):
        select_preregistered_local_cases(
            fixed,
            development_start="2023-01-01",
            development_end="2025-06-30",
        )
