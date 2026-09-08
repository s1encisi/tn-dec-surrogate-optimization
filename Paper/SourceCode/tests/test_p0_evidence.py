from __future__ import annotations

import numpy as np
import pandas as pd

from taici.p0_evidence import (
    OuterFold,
    build_joint_residual_library,
    make_inner_temporal_folds,
    make_outer_temporal_split,
    propagate_proxy_uncertainty,
)


def test_temporal_splits_are_forward_purged_and_disjoint() -> None:
    dates = pd.date_range("2023-01-01", periods=500, freq="D")
    fold = OuterFold(
        name="diagnostic",
        train_end=pd.Timestamp("2023-12-31"),
        test_start=pd.Timestamp("2024-01-01"),
        test_end=pd.Timestamp("2024-03-31"),
    )
    train, test, audit = make_outer_temporal_split(dates, fold, purge_days=3)

    assert not np.intersect1d(train, test).size
    assert dates[train].max() == pd.Timestamp("2023-12-28")
    assert dates[test].min() == pd.Timestamp("2024-01-01")
    assert audit["purged_calendar_days"] == 3
    assert audit["strictly_forward"]

    inner, validation, assignments = make_inner_temporal_folds(
        dates,
        train,
        n_splits=3,
        validation_rows=30,
        purge_days=3,
    )
    assert len(inner) == 3
    assert len(validation) == 90
    assert len(np.unique(validation)) == len(validation)
    for fit_indices, validation_indices in inner:
        assert dates[fit_indices].max() < dates[validation_indices].min()
        assert (dates[validation_indices].min() - dates[fit_indices].max()).days >= 4
    assert set(assignments["role"]) == {"inner_train", "inner_validation"}


def _outer_prediction_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dates = pd.date_range("2025-01-01", periods=40, freq="D")
    for seed in (11, 23):
        for target, model, scale in (
            ("TN_out", "Ensemble_Huber", 0.2),
            ("DEC", "ExtraTrees", 100.0),
        ):
            for position, date in enumerate(dates):
                actual = 8.0 if target == "TN_out" else 8_000.0
                residual = scale * np.sin(position / 3.0) + seed / 10_000.0
                rows.append(
                    {
                        "target": target,
                        "training_window": "2023_2025",
                        "seed": seed,
                        "model": model,
                        "Date": date,
                        "actual": actual,
                        "prediction": actual - residual,
                    }
                )
    return pd.DataFrame.from_records(rows)


def _failure_fixture() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method in ("PPO", "SAC", "TD3"):
        for episode_id in range(1, 5):
            rows.append(
                {
                    "method": method,
                    "training_seed": 11,
                    "preference_TN": 0.5,
                    "episode_id": episode_id,
                    "start_date": pd.Timestamp("2025-07-01")
                    + pd.Timedelta(days=7 * (episode_id - 1)),
                    "end_date": pd.Timestamp("2025-07-07")
                    + pd.Timedelta(days=7 * (episode_id - 1)),
                    "delta_TN_vs_RandomFeasible": 0.01,
                    "delta_DEC_vs_RandomFeasible": 5.0,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_proxy_uncertainty_is_joint_reproducible_and_non_authorizing() -> None:
    residuals = build_joint_residual_library(
        _outer_prediction_fixture(), training_window="2023_2025"
    )
    assert len(residuals) == 40
    assert {"residual_TN", "residual_DEC"}.issubset(residuals.columns)

    first = propagate_proxy_uncertainty(
        _failure_fixture(),
        residuals,
        correlations=(0.0, 0.5, 0.9),
        primary_correlation=0.5,
        replicates=1_000,
        seed=20260829,
        interval_level=0.95,
    )
    second = propagate_proxy_uncertainty(
        _failure_fixture(),
        residuals,
        correlations=(0.0, 0.5, 0.9),
        primary_correlation=0.5,
        replicates=1_000,
        seed=20260829,
        interval_level=0.95,
    )
    pd.testing.assert_frame_equal(first.method_summary, second.method_summary)
    assert len(first.method_summary) == 9
    assert len(first.episode_intervals) == 36
    assert not first.method_summary["plant_control_claim_authorized"].any()
    assert not first.episode_intervals[
        "calibrated_coverage_claim_authorized"
    ].any()
