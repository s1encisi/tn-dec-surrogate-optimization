from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tomllib

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT.parent


def _load_paired_runner():
    path = PROJECT_ROOT / "scripts" / "run_paired_random_windows.py"
    spec = importlib.util.spec_from_file_location("paired_config_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toml(relative_path: str) -> dict:
    with (PROJECT_ROOT / relative_path).open("rb") as handle:
        return tomllib.load(handle)


def _initial_manifest() -> dict:
    path = PAPER_ROOT / "InitialData" / "initial_dataset_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_formal_feature_registry_matches_every_active_contract() -> None:
    runner = _load_paired_runner()
    registry = _toml("configs/feature_registry.toml")
    paired = _toml("configs/paired_random_windows.toml")
    manifest = _initial_manifest()

    runner._validate_feature_registry(registry, paired, manifest)

    assert tuple(registry["targets"]["TN_out"]["predictors"]) == runner.TN_FEATURES
    assert tuple(registry["targets"]["DEC"]["predictors"]) == runner.DEC_FEATURES
    assert "TN_in" in registry["targets"]["TN_out"]["predictors"]
    assert "TN" not in registry["targets"]["TN_out"]["predictors"]
    for section, value in registry.items():
        if section != "historical_aliases":
            assert "past3_mean_" not in repr(value)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("old_tn_name", "predictors for TN_out"),
        ("wrong_alias", "Historical aliases"),
        ("old_feature_version", "version differs"),
        ("unexpected_table", "top-level tables"),
    ],
)
def test_feature_registry_mismatch_is_fail_closed(mutation: str, message: str) -> None:
    runner = _load_paired_runner()
    registry = deepcopy(_toml("configs/feature_registry.toml"))
    paired = _toml("configs/paired_random_windows.toml")
    manifest = _initial_manifest()

    if mutation == "old_tn_name":
        registry["targets"]["TN_out"]["predictors"][5] = "TN"
    elif mutation == "wrong_alias":
        registry["historical_aliases"]["past3_mean_TN"] = "TN"
    elif mutation == "old_feature_version":
        registry["registry"]["feature_version"] = "F_A"
    elif mutation == "unexpected_table":
        registry["versions"] = {"F_A": "legacy"}
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(RuntimeError, match=message):
        runner._validate_feature_registry(registry, paired, manifest)


def test_final_workflow_marks_old_rl_as_legacy_only() -> None:
    config = _toml("configs/final_workflow.toml")
    formal_path = "configs/formal_rl_rerun.toml"

    assert config["inputs"]["formal_rl_config"] == formal_path
    assert (PROJECT_ROOT / formal_path).is_file()
    assert "rl" not in config
    legacy = config["legacy_rl_non_manuscript"]
    assert legacy["evidence_role"] == "legacy_non_manuscript_reproduction_only"
    assert legacy["manuscript_evidence_authorized"] is False
    assert legacy["superseded_by"] == formal_path

    legacy_runner = (PROJECT_ROOT / "scripts" / "run_surrogate_rl.py").read_text(
        encoding="utf-8"
    )
    assert "_validated_legacy_rl_config(config)" in legacy_runner
    assert 'config.get("legacy_rl_non_manuscript")' in legacy_runner
    assert '"legacy_full_reproduction"' in legacy_runner
    assert '"formal_policy_matrix_expected"' not in legacy_runner


def test_matplotlib_is_a_declared_runtime_dependency() -> None:
    project = _toml("pyproject.toml")
    assert "matplotlib>=3.10,<4.0" in project["project"]["dependencies"]
