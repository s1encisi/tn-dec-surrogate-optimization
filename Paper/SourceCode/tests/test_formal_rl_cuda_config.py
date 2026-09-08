from __future__ import annotations

import importlib.util
from pathlib import Path
import tomllib

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_runner():
    path = PROJECT_ROOT / "scripts" / "run_formal_rl_rerun.py"
    spec = importlib.util.spec_from_file_location("formal_rl_runner_for_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_config_requires_cuda_with_sub_90_percent_guard() -> None:
    config_path = PROJECT_ROOT / "configs" / "formal_rl_rerun.toml"
    with config_path.open("rb") as handle:
        rl = tomllib.load(handle)["rl"]
    assert rl["device"] == "cuda:0"
    assert 0.0 < rl["cuda_allocator_memory_fraction"] <= 0.40
    assert rl["cuda_allocator_memory_fraction"] < rl["cuda_total_memory_abort_fraction"]
    assert rl["cuda_total_memory_abort_fraction"] <= 0.85
    assert rl["cuda_total_memory_abort_fraction"] < rl["cuda_hard_memory_fraction"] <= 0.90
    assert 0.0 < rl["cuda_preflight_max_global_fraction"] < rl["cuda_total_memory_abort_fraction"]
    assert rl["cuda_non_allocator_reserve_fraction"] >= 0.05
    assert 0.0 < rl["cuda_watchdog_interval_seconds"] <= 1.0
    assert rl["cuda_memory_check_interval_steps"] > 0


def test_compute_configuration_keeps_cpu_fallback_explicit() -> None:
    runner = _load_runner()
    audit = runner._configure_compute({"device": "cpu"})
    assert audit["status"] == "PASSED"
    assert audit["device_type"] == "cpu"
    assert audit["memory_guard_active"] is False


def test_cuda_compute_configuration_uses_protected_real_device_when_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable on this test host.")
    runner = _load_runner()
    config_path = PROJECT_ROOT / "configs" / "formal_rl_rerun.toml"
    with config_path.open("rb") as handle:
        rl = tomllib.load(handle)["rl"]
    audit = runner._configure_compute(rl)
    assert audit["device_type"] == "cuda"
    assert audit["device_name"]
    assert audit["allocator_memory_fraction_cap"] <= 0.40
    assert audit["global_memory_abort_fraction"] == 0.85
    assert audit["global_memory_hard_fraction"] == 0.90
    assert audit["initial_memory"]["global_used_fraction"] < 0.85


def test_gpu_guard_summary_rejects_abort_boundary() -> None:
    runner = _load_runner()
    audit = {
        "device_type": "cuda",
        "global_memory_abort_fraction": 0.85,
        "global_memory_hard_fraction": 0.90,
    }
    history = [
        {
            "algorithm": "PPO",
            "training_seed": 11,
            "global_used_fraction": 0.85,
        }
    ]
    with pytest.raises(RuntimeError, match="GPU memory guard failed"):
        runner._gpu_guard_summary(history, audit, expected_pairs=[("PPO", 11)])
