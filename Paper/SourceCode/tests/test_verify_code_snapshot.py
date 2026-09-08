from pathlib import Path

import pytest

from verify_code_snapshot import role, source_for


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/formal_rl_rerun.toml",
        "scripts/run_formal_rl_rerun.py",
        "tests/test_formal_rl_cuda_config.py",
        "tests/test_formal_rl_locking.py",
    ],
)
def test_formal_rl_files_are_recorded_as_generated_within_snapshot(
    relative_path: str,
) -> None:
    category, source = source_for(Path(relative_path))

    assert category == "generated_for_formal_rl_rerun"
    assert source is None


@pytest.mark.parametrize(
    "relative_path",
    [
        "plotting/generate_paper_assets_v3.py",
        "plotting/build_plotting_workbooks.mjs",
        "README_reproducibility.md",
        "verify_code_snapshot.py",
        "tests/test_configuration_contracts.py",
        "tests/test_verify_code_snapshot.py",
    ],
)
def test_paper_support_files_are_recorded_as_generated_within_snapshot(
    relative_path: str,
) -> None:
    category, source = source_for(Path(relative_path))

    assert category == "generated_for_paper"
    assert source is None


@pytest.mark.parametrize(
    "relative_path",
    [
        "configs/feature_registry.toml",
        "configs/final_workflow.toml",
        "pyproject.toml",
        "scripts/run_paired_random_windows.py",
        "scripts/run_surrogate_rl.py",
    ],
)
def test_compatibility_changes_are_recorded_as_adapted(
    relative_path: str,
) -> None:
    category, source = source_for(Path(relative_path))

    assert category == "adapted_for_frozen_initial_dataset"
    assert source is not None


@pytest.mark.parametrize(
    "relative_path",
    [
        "tests/test_formal_rl_cuda_config.py",
        "tests/test_formal_rl_locking.py",
        "tests/test_surrogate_rl.py",
        "tests/test_explainability.py",
    ],
)
def test_test_files_keep_verification_role_before_topic_matching(
    relative_path: str,
) -> None:
    assert role(Path(relative_path)) == "verification_test"
