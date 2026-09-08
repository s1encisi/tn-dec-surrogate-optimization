from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.initial_dataset import (  # noqa: E402
    DEC_FEATURES,
    EXPECTED_COLUMNS,
    InitialDatasetError,
    TN_FEATURES,
    load_initial_dataset,
)
from taici.surrogate_rl import _daily_action_history  # noqa: E402


INITIAL_DATASET = PROJECT_ROOT.parent / "InitialData" / "initial_dataset.csv"
EXPECTED_SHA256 = "f65dddf4f4a7137a7fa3f48ca6442788bb665449405b5b24bdb18732e28bd692"


def test_frozen_initial_dataset_contract() -> None:
    initial = load_initial_dataset(INITIAL_DATASET)

    assert initial.file_sha256 == EXPECTED_SHA256
    assert initial.frame.shape == (1_076, 25)
    assert tuple(initial.frame.columns) == EXPECTED_COLUMNS
    assert tuple(initial.feature_set.bundles["TN_out"].feature_names.values())[0] == TN_FEATURES
    assert tuple(initial.feature_set.bundles["DEC"].feature_names.values())[0] == DEC_FEATURES
    assert not any(column.startswith("past3_mean_") for column in initial.frame.columns)
    assert {"PPA_current_observed", "DO_current_observed"}.isdisjoint(TN_FEATURES)
    assert {"PPA_current_observed", "DO_current_observed"}.isdisjoint(DEC_FEATURES)


def test_manifest_hash_is_fail_closed(tmp_path: Path) -> None:
    copied_data = tmp_path / INITIAL_DATASET.name
    copied_manifest = tmp_path / "initial_dataset_manifest.json"
    shutil.copyfile(INITIAL_DATASET, copied_data)
    shutil.copyfile(INITIAL_DATASET.with_name(copied_manifest.name), copied_manifest)
    copied_data.write_bytes(copied_data.read_bytes() + b"\n")

    with pytest.raises(InitialDatasetError, match="file hash differs"):
        load_initial_dataset(copied_data)


def test_rl_audit_columns_reproduce_frozen_action_rate() -> None:
    initial = load_initial_dataset(INITIAL_DATASET)
    history = _daily_action_history(initial.frame)
    training = history.loc["2023-01-01":"2024-12-31"]

    assert len(training) == 731
    observed = training.diff().abs().quantile(0.90).to_numpy(float)
    np.testing.assert_allclose(observed, np.array([216.187, 0.573]), rtol=0.0, atol=1e-12)


def test_formal_entries_do_not_rebuild_initial_inputs() -> None:
    entries = (
        "run_paired_random_windows.py",
        "run_final_proxy_refit.py",
        "run_final_xai.py",
        "run_surrogate_rl.py",
    )
    forbidden = ("build_paired_hrt3_features", "modeling_data_v1", "past3_mean_")
    for entry in entries:
        text = (PROJECT_ROOT / "scripts" / entry).read_text(encoding="utf-8")
        assert "load_initial_dataset" in text
        for token in forbidden:
            assert token not in text
