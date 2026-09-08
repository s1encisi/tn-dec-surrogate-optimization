from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = PROJECT_ROOT / "scripts" / "run_formal_rl_rerun.py"
    spec = importlib.util.spec_from_file_location("formal_rl_locking_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runner():
    return _load_runner()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _make_contract(runner, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SimpleNamespace:
    run_dir = tmp_path / "formal_run"
    (run_dir / "provenance").mkdir(parents=True)
    (run_dir / "provenance" / "formal_rl_rerun.toml").write_text(
        "# isolated unit-test placeholder\n", encoding="utf-8"
    )

    hashes = {
        "config": "a" * 64,
        "initial_dataset": "b" * 64,
        "initial_dataset_manifest": "c" * 64,
        "proxy_manifest": "d" * 64,
        "proxy_artifact": "e" * 64,
        "formal_runner": "f" * 64,
        "legacy_runner_helpers": "1" * 64,
        "surrogate_environment": "2" * 64,
        "fast_proxy": "3" * 64,
    }
    snapshot_hashes = {
        name: hashes[name]
        for name in (
            "config",
            "formal_runner",
            "legacy_runner_helpers",
            "surrogate_environment",
            "fast_proxy",
        )
    }

    selections: list[dict[str, object]] = []
    for algorithm in runner.FORMAL_ALGORITHMS:
        for seed in runner.FORMAL_TRAINING_SEEDS:
            relative = Path("models") / algorithm / f"seed_{seed}" / "best_model.zip"
            model_path = run_dir / relative
            model_path.parent.mkdir(parents=True, exist_ok=True)
            model_path.write_bytes(f"{algorithm}:{seed}".encode())
            selections.append(
                {
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "model_path": relative.as_posix(),
                    "model_sha256": runner.sha256_file(model_path),
                    "best_validation_step": runner.FORMAL_INTERACTION_BUDGET,
                    "best_validation_HV": 0.5,
                }
            )

    fingerprint = runner._development_fingerprint(
        hashes, runner.FORMAL_INTERACTION_BUDGET, selections
    )
    source_lock = {
        "active_source_and_input_hashes": hashes,
        "provenance_snapshot_hashes": snapshot_hashes,
    }
    source_lock_path = run_dir / "source_input_lock_at_start.json"
    _write_json(source_lock_path, source_lock)
    _write_json(run_dir / "frozen_validation_selections.json", selections)

    status = {
        "status": "DEVELOPMENT_COMPLETE_TEST_SEALED",
        "test_accessed": False,
        "test_authorized": False,
        "fingerprint": fingerprint,
    }
    manifest = {
        "stage": "formal_rl_development",
        "status": "completed_test_sealed",
        "algorithms": list(runner.FORMAL_ALGORITHMS),
        "training_seeds": list(runner.FORMAL_TRAINING_SEEDS),
        "common_interaction_budget_per_algorithm_seed": runner.FORMAL_INTERACTION_BUDGET,
        "policy_models_completed": 15,
        "fingerprint": fingerprint,
        "source_and_input_hashes": hashes,
        "source_input_lock_at_start_sha256": runner.sha256_file(source_lock_path),
    }
    authorization = {
        "schema_version": "formal_rl_budget_freeze_v1",
        "status": "FORMAL_DEVELOPMENT_FROZEN_TEST_AUTHORIZED",
        "test_authorized": True,
        "repeat_test_access_authorized": False,
        "test_accessed_at_freeze": False,
        "selected_budget": runner.FORMAL_INTERACTION_BUDGET,
        "development_fingerprint": fingerprint,
        "frozen_model_hashes": [
            {
                "algorithm": row["algorithm"],
                "training_seed": row["training_seed"],
                "model_sha256": row["model_sha256"],
            }
            for row in selections
        ],
    }
    config = {
        "inputs": {
            "initial_dataset": str(tmp_path / "initial_dataset.csv"),
            "proxy_run": str(tmp_path / "proxy_run"),
        },
        "rl": {
            "algorithms": list(runner.FORMAL_ALGORITHMS),
            "training_seeds": list(runner.FORMAL_TRAINING_SEEDS),
            "extension_max_timesteps_per_algorithm_seed": runner.FORMAL_INTERACTION_BUDGET,
            "device": "cuda:0",
        },
    }
    monkeypatch.setattr(runner, "load_toml", lambda _path: config)
    monkeypatch.setattr(
        runner,
        "_source_hashes",
        lambda _config, _initial, _proxy: dict(hashes),
    )
    monkeypatch.setattr(
        runner,
        "_provenance_snapshot_hashes",
        lambda _run_dir: dict(snapshot_hashes),
    )
    return SimpleNamespace(
        run_dir=run_dir,
        hashes=hashes,
        snapshot_hashes=snapshot_hashes,
        selections=selections,
        status=status,
        manifest=manifest,
        authorization=authorization,
        config=config,
    )


def _install_selections(runner, case: SimpleNamespace, selections: list[dict[str, object]]) -> None:
    _write_json(case.run_dir / "frozen_validation_selections.json", selections)
    fingerprint = runner._development_fingerprint(
        case.hashes, runner.FORMAL_INTERACTION_BUDGET, selections
    )
    case.status["fingerprint"] = fingerprint
    case.manifest["fingerprint"] = fingerprint
    case.authorization["development_fingerprint"] = fingerprint
    case.authorization["frozen_model_hashes"] = [
        {
            "algorithm": row["algorithm"],
            "training_seed": row["training_seed"],
            "model_sha256": row["model_sha256"],
        }
        for row in selections
    ]


def _validate(runner, case: SimpleNamespace):
    return runner._validate_locked_test_contract(
        case.run_dir,
        case.status,
        case.authorization,
        case.manifest,
    )


def test_development_fingerprint_is_sensitive_to_selection_order_and_content(runner) -> None:
    hashes = {"config": "a" * 64, "formal_runner": "b" * 64}
    selections = [
        {"algorithm": "PPO", "training_seed": 11, "model_sha256": "1" * 64},
        {"algorithm": "SAC", "training_seed": 23, "model_sha256": "2" * 64},
    ]
    original = runner._development_fingerprint(hashes, 32_000, selections)
    reordered = runner._development_fingerprint(hashes, 32_000, list(reversed(selections)))
    changed = deepcopy(selections)
    changed[0]["model_sha256"] = "9" * 64
    changed_content = runner._development_fingerprint(hashes, 32_000, changed)

    assert original != reordered
    assert original != changed_content


def test_smoke_stage_is_rejected_before_test_access(runner, monkeypatch, tmp_path: Path) -> None:
    case = _make_contract(runner, monkeypatch, tmp_path)
    case.manifest["stage"] = "formal_rl_development_smoke"

    with pytest.raises(RuntimeError, match="Smoke or non-formal"):
        _validate(runner, case)


@pytest.mark.parametrize("changed_key", ["config", "formal_runner"])
def test_changed_config_or_source_hash_is_rejected(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    changed_key: str,
) -> None:
    case = _make_contract(runner, monkeypatch, tmp_path)
    changed_hashes = dict(case.hashes)
    changed_hashes[changed_key] = "9" * 64
    monkeypatch.setattr(
        runner,
        "_source_hashes",
        lambda _config, _initial, _proxy: changed_hashes,
    )

    with pytest.raises(RuntimeError, match="Frozen source/input hashes changed"):
        _validate(runner, case)


def test_missing_selection_is_rejected(runner, monkeypatch, tmp_path: Path) -> None:
    case = _make_contract(runner, monkeypatch, tmp_path)
    selections = deepcopy(case.selections[:-1])
    _install_selections(runner, case, selections)

    with pytest.raises(RuntimeError, match="exactly 15 records"):
        _validate(runner, case)


def test_duplicate_selection_is_rejected(runner, monkeypatch, tmp_path: Path) -> None:
    case = _make_contract(runner, monkeypatch, tmp_path)
    selections = deepcopy(case.selections)
    selections[-1] = deepcopy(selections[0])
    _install_selections(runner, case, selections)

    with pytest.raises(RuntimeError, match="Invalid or duplicate frozen selection"):
        _validate(runner, case)


def test_selection_path_escape_is_rejected(runner, monkeypatch, tmp_path: Path) -> None:
    case = _make_contract(runner, monkeypatch, tmp_path)
    selections = deepcopy(case.selections)
    selections[0]["model_path"] = "../outside.zip"
    _install_selections(runner, case, selections)

    with pytest.raises(RuntimeError, match="registered layout"):
        _validate(runner, case)


def test_watchdog_stop_records_violation_without_throwing(runner, monkeypatch) -> None:
    snapshot = {
        "device_index": 0,
        "total_bytes": 100,
        "free_bytes": 14,
        "global_used_bytes": 86,
        "global_used_fraction": 0.86,
        "process_allocated_bytes": 10,
        "process_reserved_bytes": 12,
    }
    monkeypatch.setattr(runner, "_cuda_memory_snapshot", lambda _device: dict(snapshot))
    watchdog = runner.CudaMemoryWatchdog(
        {
            "device_type": "cuda",
            "device_index": 0,
            "global_memory_abort_fraction": 0.85,
            "global_memory_hard_fraction": 0.90,
            "watchdog_interval_seconds": 0.01,
        },
        phase="unit_test",
    )

    watchdog.stop()

    assert watchdog.history[-1]["event"] == "watchdog_stop"
    with pytest.raises(RuntimeError, match="stop sample reached"):
        watchdog.assert_safe()


class _ExplodingTargetBundle:
    @property
    def anchor(self):
        raise RuntimeError("locked panel sentinel")


def test_committed_test_failure_updates_access_and_run_status(
    runner,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "formal_run"
    run_dir.mkdir()
    fingerprint = "f" * 64
    _write_json(run_dir / "run_status.json", {"status": "sealed"})
    _write_json(run_dir / "budget_freeze.json", {"authorized": True})
    _write_json(run_dir / "development_manifest.json", {"fingerprint": fingerprint})
    _write_json(run_dir / "scenario_fit_lock.json", {})
    config = {"output": {"encoding": "utf-8"}}
    rl_config = {"test_start": "2025-07-01", "test_end": "2025-12-31"}
    initial = SimpleNamespace(
        feature_set=SimpleNamespace(bundles={"TN_out": _ExplodingTargetBundle()}),
        frame=object(),
    )

    monkeypatch.setattr(
        runner,
        "_validate_locked_test_contract",
        lambda *_args: (config, rl_config, tmp_path / "initial.csv", tmp_path / "proxy", []),
    )
    monkeypatch.setattr(
        runner,
        "_configure_compute",
        lambda _config: {"status": "PASSED", "device_type": "cpu"},
    )
    monkeypatch.setattr(
        runner,
        "_preflight_frozen_models",
        lambda *_args: (
            [],
            [],
            {"status": "NOT_APPLICABLE_CPU"},
            {},
            SimpleNamespace(
                history=[],
                request_stop=lambda: None,
                stop=lambda: None,
                assert_safe=lambda: None,
            ),
        ),
    )
    monkeypatch.setattr(runner, "load_initial_dataset", lambda _path: initial)
    monkeypatch.setattr(runner, "_load_proxy", lambda _path: (object(), _path, _path))
    monkeypatch.setattr(runner.legacy, "_limit_estimator_threads", lambda _bundle: {})
    monkeypatch.setattr(
        runner.FastExactProxyBundle,
        "from_frozen",
        staticmethod(lambda _bundle: object()),
    )
    monkeypatch.setattr(runner, "build_scenario_data", lambda *_args: object())
    monkeypatch.setattr(runner, "_scenario_fit_lock", lambda _scenario: {})

    with pytest.raises(RuntimeError, match="locked panel sentinel"):
        runner._test_once(argparse.Namespace(run_dir=str(run_dir)))

    access = json.loads((run_dir / "test_once" / "TEST_ACCESS.json").read_text())
    status = json.loads((run_dir / "run_status.json").read_text())
    assert access["status"] == "TEST_ACCESS_FAILED_NO_RETRY_ALLOWED"
    assert access["test_accessed"] is True
    assert access["repeat_access_allowed"] is False
    assert status["status"] == "FORMAL_RL_TEST_FAILED_NO_RETRY_ALLOWED"
    assert status["test_accessed"] is True
    assert status["test_access_attempts"] == 1
