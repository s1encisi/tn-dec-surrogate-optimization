from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.config import load_toml  # noqa: E402
from taici.initial_dataset import load_initial_dataset  # noqa: E402
from taici.paired_random_windows import (  # noqa: E402
    BASELINES,
    BASE_MODELS,
    FEATURE_KEY,
    OUTER_SEEDS,
    TARGETS,
    _indices_for_window,
    build_paired_model_registry,
    make_shared_2025_assignments,
)


@pytest.fixture(scope="module")
def paired_inputs():
    config = load_toml("configs/paired_random_windows.toml")
    initial = load_initial_dataset(
        PROJECT_ROOT.parent / "InitialData" / "initial_dataset.csv"
    )
    return config, initial.frame, initial.feature_set


def test_strict_common_panel_counts_and_shared_target_dates(paired_inputs) -> None:
    _, _, feature_set = paired_inputs
    counts = pd.Series(feature_set.common_dates.year).value_counts().sort_index().to_dict()
    assert counts == {2023: 362, 2024: 366, 2025: 348}
    assert feature_set.common_dates[feature_set.common_dates.year == 2025].min() == pd.Timestamp(
        "2025-01-04"
    )
    tn_dates = pd.DatetimeIndex(feature_set.bundles["TN_out"].anchor["Date"])
    dec_dates = pd.DatetimeIndex(feature_set.bundles["DEC"].anchor["Date"])
    assert tn_dates.equals(dec_dates)


def test_exact_feature_contract_and_no_outcome_history(paired_inputs) -> None:
    _, _, feature_set = paired_inputs
    expected_tn = (
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
    expected_dec = (
        "doy_sin",
        "doy_cos",
        "time_index_days",
        "Q",
        "COD",
        "NH3N",
        "T",
        "PPA",
        "DO",
        "MLSS",
    )
    assert feature_set.bundles["TN_out"].feature_names[FEATURE_KEY] == expected_tn
    assert feature_set.bundles["DEC"].feature_names[FEATURE_KEY] == expected_dec
    for names in (expected_tn, expected_dec):
        assert not any(name.startswith("lag") or "rolling" in name.lower() for name in names)
        assert not any(name.lower().endswith("_out") for name in names)
        assert not {"DEC", "TEC", "LTDEC", "PPD"}.intersection(names)


def test_absolute_time_anchor_and_materialized_queue_consistency(paired_inputs) -> None:
    _, data, feature_set = paired_inputs
    bundle = feature_set.bundles["TN_out"]
    row = bundle.anchor.index[bundle.anchor["Date"].eq(pd.Timestamp("2025-01-04"))]
    assert len(row) == 1
    features = bundle.matrix(FEATURE_KEY).iloc[int(row[0])]
    assert features["time_index_days"] == 734.0
    source_row = data.loc[data["Date"].eq(pd.Timestamp("2025-01-04"))].iloc[0]
    expected = source_row[
        ["PPA_older", "PPA_recent", "PPA_historical_action"]
    ].mean()
    assert features["PPA"] == pytest.approx(float(expected))


def test_shared_ordinary_random_assignments_and_window_counts(paired_inputs) -> None:
    _, _, feature_set = paired_inputs
    first = make_shared_2025_assignments(feature_set.common_dates)
    second = make_shared_2025_assignments(feature_set.common_dates)
    pd.testing.assert_frame_equal(first, second)
    assert tuple(sorted(first["seed"].unique())) == tuple(sorted(OUTER_SEEDS))
    counts = first.groupby(["seed", "role"], observed=True).size().unstack()
    assert counts["outer_train"].eq(278).all()
    assert counts["outer_test"].eq(70).all()
    assert first["ordinary_daily_random"].all()

    for target in TARGETS:
        bundle = feature_set.bundles[target]
        for seed in OUTER_SEEDS:
            assignment = first.loc[first["seed"].eq(seed)]
            train_2025, test_2025 = _indices_for_window(bundle, assignment, "2025_only")
            train_three_year, test_three_year = _indices_for_window(
                bundle, assignment, "2023_2025"
            )
            assert len(train_2025) == 278
            assert len(train_three_year) == 1006
            assert len(test_2025) == len(test_three_year) == 70
            np.testing.assert_array_equal(test_2025, test_three_year)
            assert np.intersect1d(train_2025, test_2025).size == 0
            assert np.intersect1d(train_three_year, test_three_year).size == 0


def test_exact_15_model_registry_and_candidate_parameters(paired_inputs) -> None:
    config, _, feature_set = paired_inputs
    registry = build_paired_model_registry(
        seed=11,
        n_features=len(feature_set.bundles["TN_out"].feature_names[FEATURE_KEY]),
        n_jobs=1,
        candidate_count=int(config["models"]["candidate_count"]),
        tabnet_config=config["tabnet"],
    )
    assert tuple(registry) == (*BASE_MODELS, *BASELINES)
    assert tuple(config["models"]["base_pool"]) == BASE_MODELS
    assert tuple(config["ensemble"]["eligible_base_models"]) == BASE_MODELS
    assert len(registry["TabNet"].candidates) == 2
    assert list(registry["TabNet"].estimator.named_steps) == ["model"]
    for model, spec in registry.items():
        assert spec.feature_key == FEATURE_KEY
        for candidate in spec.candidates:
            clone(spec.estimator).set_params(**candidate)
        if model in BASE_MODELS:
            assert spec.ensemble_eligible
        else:
            assert not spec.ensemble_eligible
