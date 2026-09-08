from __future__ import annotations

import argparse
import _thread
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import threading
import time
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for candidate in (SRC_ROOT, SCRIPT_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from pymoo.indicators.hv import HV  # noqa: E402
from pymoo.indicators.igd_plus import IGDPlus  # noqa: E402
from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402

import run_surrogate_rl as legacy  # noqa: E402
from taici.config import load_toml  # noqa: E402
from taici.fast_proxy import FastExactProxyBundle, audit_fast_proxy_fidelity  # noqa: E402
from taici.final_proxy import FinalProxyBundle  # noqa: E402
from taici.initial_dataset import _as_feature_set, load_initial_dataset  # noqa: E402
from taici.io import sha256_file, write_json  # noqa: E402
from taici.paired_random_windows import FEATURE_KEY  # noqa: E402
from taici.surrogate_rl import (  # noqa: E402
    ScenarioData,
    algorithm_multiobjective_summary,
    benchmark_proxy_step,
    build_scenario_data,
    non_dominated_mask,
    rollout_historical_baseline,
    rollout_policy,
    rollout_simple_baseline,
    summarize_episodes,
)


ALGORITHM_CLASSES = {"PPO": PPO, "SAC": SAC, "TD3": TD3}
BASELINE_METHODS = ("HistoricalObserved", "KeepPrevious", "RandomFeasible")
FORMAL_ALGORITHMS = ("PPO", "SAC", "TD3")
FORMAL_TRAINING_SEEDS = (11, 23, 37, 53, 71)
FORMAL_INTERACTION_BUDGET = 32_000


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only development and one-shot locked evaluation for the formal "
            "preference-conditioned PPO/SAC/TD3 rerun."
        )
    )
    parser.add_argument("--phase", required=True, choices=("develop", "test"))
    parser.add_argument("--config", default="configs/formal_rl_rerun.toml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Development-only common interaction budget; must equal a registered budget.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run PPO/seed 11 for 1,000 development steps; never opens the locked panel.",
    )
    return parser.parse_args()


def _resolve(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _cuda_memory_snapshot(device_index: int) -> dict[str, Any]:
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    used_bytes = int(total_bytes - free_bytes)
    return {
        "device_index": int(device_index),
        "total_bytes": int(total_bytes),
        "free_bytes": int(free_bytes),
        "global_used_bytes": used_bytes,
        "global_used_fraction": float(used_bytes / total_bytes),
        "process_allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "process_reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
    }


def _configure_compute(rl_config: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed unless the requested CUDA device and memory guard are available."""

    requested = str(rl_config["device"]).strip().lower()
    device = torch.device(requested)
    audit: dict[str, Any] = {
        "status": "PASSED",
        "requested_device": requested,
        "torch_version": str(torch.__version__),
        "torch_compiled_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if device.type != "cuda":
        audit.update({"device_type": "cpu", "memory_guard_active": False})
        return audit
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this Python interpreter has no usable CUDA-enabled PyTorch."
        )

    device_index = 0 if device.index is None else int(device.index)
    if device_index < 0 or device_index >= torch.cuda.device_count():
        raise RuntimeError(f"Requested CUDA device index is unavailable: {device_index}.")
    requested_allocator_fraction = float(rl_config["cuda_allocator_memory_fraction"])
    abort_fraction = float(rl_config["cuda_total_memory_abort_fraction"])
    hard_fraction = float(rl_config["cuda_hard_memory_fraction"])
    preflight_fraction = float(rl_config["cuda_preflight_max_global_fraction"])
    reserve_fraction = float(rl_config["cuda_non_allocator_reserve_fraction"])
    watchdog_interval = float(rl_config["cuda_watchdog_interval_seconds"])
    check_interval = int(rl_config["cuda_memory_check_interval_steps"])
    if not 0.0 < requested_allocator_fraction <= 0.40:
        raise RuntimeError("CUDA allocator fraction must be in (0, 0.40].")
    if not requested_allocator_fraction < abort_fraction <= 0.85:
        raise RuntimeError("CUDA total-memory abort fraction must be at most 0.85.")
    if not abort_fraction < hard_fraction <= 0.90:
        raise RuntimeError("CUDA hard-memory fraction must be above abort and at most 0.90.")
    if not 0.0 < preflight_fraction < abort_fraction:
        raise RuntimeError("CUDA preflight fraction must be below the abort threshold.")
    if not 0.05 <= reserve_fraction < hard_fraction:
        raise RuntimeError("CUDA non-allocator reserve fraction must be at least 0.05.")
    if not 0.0 < watchdog_interval <= 1.0:
        raise RuntimeError("CUDA watchdog interval must be in (0, 1] seconds.")
    if check_interval <= 0:
        raise RuntimeError("CUDA memory check interval must be positive.")

    torch.cuda.set_device(device_index)
    before_allocator = _cuda_memory_snapshot(device_index)
    if before_allocator["global_used_fraction"] >= preflight_fraction:
        raise RuntimeError(
            "CUDA global memory use is too high for a protected training start."
        )
    dynamic_allocator_ceiling = (
        hard_fraction
        - float(before_allocator["global_used_fraction"])
        - reserve_fraction
    )
    allocator_fraction = min(requested_allocator_fraction, dynamic_allocator_ceiling)
    if allocator_fraction <= 0.05:
        raise RuntimeError("Insufficient protected CUDA memory remains for this run.")
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device_index)
    applied_allocator_fraction = float(
        torch.cuda.get_per_process_memory_fraction(device_index)
    )
    if applied_allocator_fraction > allocator_fraction + 1e-9:
        raise RuntimeError("CUDA allocator cap readback exceeds the protected limit.")
    probe = torch.ones((256, 256), dtype=torch.float32, device=device)
    _ = probe @ probe
    torch.cuda.synchronize(device_index)
    del probe
    initial = _cuda_memory_snapshot(device_index)
    if initial["global_used_fraction"] >= abort_fraction:
        raise RuntimeError(
            "CUDA global memory use already exceeds the configured safety threshold."
        )
    audit.update(
        {
            "device_type": "cuda",
            "device_index": device_index,
            "device_name": torch.cuda.get_device_name(device_index),
            "device_capability": list(torch.cuda.get_device_capability(device_index)),
            "requested_allocator_memory_fraction_cap": requested_allocator_fraction,
            "allocator_memory_fraction_cap": applied_allocator_fraction,
            "global_memory_abort_fraction": abort_fraction,
            "global_memory_hard_fraction": hard_fraction,
            "preflight_max_global_fraction": preflight_fraction,
            "non_allocator_reserve_fraction": reserve_fraction,
            "watchdog_interval_seconds": watchdog_interval,
            "memory_check_interval_steps": check_interval,
            "memory_guard_active": True,
            "memory_before_allocator_cap": before_allocator,
            "initial_memory": initial,
        }
    )
    return audit


class CudaMemoryWatchdog:
    """Continuously interrupt the main thread before global VRAM reaches 90%."""

    def __init__(self, compute_audit: Mapping[str, Any], *, phase: str) -> None:
        self.enabled = compute_audit.get("device_type") == "cuda"
        self.device_index = int(compute_audit.get("device_index", 0))
        self.abort_fraction = float(
            compute_audit.get("global_memory_abort_fraction", 1.0)
        )
        self.hard_fraction = float(
            compute_audit.get("global_memory_hard_fraction", 1.0)
        )
        self.interval_seconds = float(
            compute_audit.get("watchdog_interval_seconds", 1.0)
        )
        self.phase = str(phase)
        self.algorithm = ""
        self.training_seed: int | str = ""
        self.history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._violation: str | None = None
        self._thread: threading.Thread | None = None
        self._stopped = False

    def set_context(self, *, algorithm: str = "", training_seed: int | str = "") -> None:
        with self._lock:
            self.algorithm = str(algorithm)
            self.training_seed = training_seed

    def _capture(self, event: str, checkpoint_step: int | None = None) -> dict[str, Any]:
        snapshot = _cuda_memory_snapshot(self.device_index)
        with self._lock:
            row = {
                "timestamp_utc": _utc_now(),
                "phase": self.phase,
                "algorithm": self.algorithm,
                "training_seed": self.training_seed,
                "checkpoint_step": checkpoint_step,
                "event": event,
                **snapshot,
                "abort_fraction": self.abort_fraction,
                "hard_fraction": self.hard_fraction,
            }
            self.history.append(row)
        return row

    def _raise_if_unsafe(self, row: Mapping[str, Any]) -> None:
        used = float(row["global_used_fraction"])
        if used >= self.abort_fraction:
            self._violation = (
                f"CUDA global memory reached {used:.4f}, at or above the "
                f"protected abort threshold {self.abort_fraction:.4f}."
            )
            raise RuntimeError(self._violation)

    def sample(self, event: str, checkpoint_step: int | None = None) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        row = self._capture(event, checkpoint_step)
        self._raise_if_unsafe(row)
        return row

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                row = self._capture("watchdog_poll")
                if float(row["global_used_fraction"]) >= self.abort_fraction:
                    self._violation = (
                        "CUDA watchdog observed global memory at or above the "
                        f"protected abort threshold: {row['global_used_fraction']:.4f}."
                    )
                    _thread.interrupt_main()
                    return
            except BaseException as exc:  # pragma: no cover - defensive hardware path
                self._violation = f"CUDA watchdog failed closed: {type(exc).__name__}: {exc}"
                _thread.interrupt_main()
                return

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self.sample("watchdog_start")
        self._thread = threading.Thread(
            target=self._run,
            name=f"cuda-memory-watchdog-{self.phase}",
            daemon=True,
        )
        self._thread.start()

    def request_stop(self) -> None:
        self._stop_event.set()

    def stop(self) -> None:
        if not self.enabled or self._stopped:
            return
        self._stopped = True
        self.request_stop()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4.0))
            if self._thread.is_alive() and self._violation is None:
                self._violation = "CUDA watchdog thread did not stop within the protected timeout."
        if self._violation is None:
            try:
                row = self._capture("watchdog_stop")
                if float(row["global_used_fraction"]) >= self.abort_fraction:
                    self._violation = (
                        "CUDA watchdog stop sample reached the protected abort threshold: "
                        f"{row['global_used_fraction']:.4f}."
                    )
            except BaseException as exc:  # pragma: no cover - defensive hardware path
                self._violation = (
                    f"CUDA watchdog stop failed closed: {type(exc).__name__}: {exc}"
                )

    def assert_safe(self) -> None:
        if self._violation is not None:
            raise RuntimeError(self._violation)


def _assert_model_device(model: Any, requested_device: str) -> str:
    expected = torch.device(requested_device)
    try:
        actual = next(model.policy.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise RuntimeError("Unable to verify the SB3 policy device.") from exc
    if actual.type != expected.type:
        raise RuntimeError(f"SB3 policy device mismatch: expected {expected}, got {actual}.")
    if expected.type == "cuda" and expected.index is not None and actual.index != expected.index:
        raise RuntimeError(f"SB3 policy CUDA index mismatch: expected {expected}, got {actual}.")
    return str(actual)


def _gpu_guard_summary(
    history: Sequence[Mapping[str, Any]],
    compute_audit: Mapping[str, Any],
    *,
    expected_pairs: Sequence[tuple[str, int]],
) -> dict[str, Any]:
    if compute_audit.get("device_type") != "cuda":
        return {
            "status": "NOT_APPLICABLE_CPU",
            "memory_guard_active": False,
            "sample_count": 0,
        }
    if not history:
        raise RuntimeError("CUDA run completed without GPU memory watchdog samples.")
    observed_pairs = {
        (str(row["algorithm"]), int(row["training_seed"]))
        for row in history
        if str(row.get("algorithm", "")) and str(row.get("training_seed", ""))
    }
    missing_pairs = sorted(set(expected_pairs) - observed_pairs)
    if missing_pairs:
        raise RuntimeError(f"GPU watchdog coverage is missing model pairs: {missing_pairs}.")
    maximum = max(float(row["global_used_fraction"]) for row in history)
    abort_fraction = float(compute_audit["global_memory_abort_fraction"])
    hard_fraction = float(compute_audit["global_memory_hard_fraction"])
    if maximum >= abort_fraction:
        raise RuntimeError(
            f"GPU memory guard failed: observed {maximum:.4f} >= {abort_fraction:.4f}."
        )
    if maximum >= hard_fraction:
        raise RuntimeError(
            f"GPU hard memory limit failed: observed {maximum:.4f} >= {hard_fraction:.4f}."
        )
    return {
        "status": "PASSED",
        "memory_guard_active": True,
        "sample_count": len(history),
        "expected_model_pairs": [
            {"algorithm": algorithm, "training_seed": seed}
            for algorithm, seed in expected_pairs
        ],
        "observed_model_pairs": [
            {"algorithm": algorithm, "training_seed": seed}
            for algorithm, seed in sorted(observed_pairs)
        ],
        "maximum_global_used_fraction": maximum,
        "abort_fraction": abort_fraction,
        "hard_fraction": hard_fraction,
        "all_samples_below_abort": True,
        "all_samples_below_hard_limit": True,
    }


def _global_gpu_guard_summary(
    history: Sequence[Mapping[str, Any]], compute_audit: Mapping[str, Any]
) -> dict[str, Any]:
    if compute_audit.get("device_type") != "cuda":
        return {
            "status": "NOT_APPLICABLE_CPU",
            "memory_guard_active": False,
            "sample_count": 0,
        }
    if not history:
        raise RuntimeError("Global CUDA guard completed without memory samples.")
    maximum = max(float(row["global_used_fraction"]) for row in history)
    abort_fraction = float(compute_audit["global_memory_abort_fraction"])
    hard_fraction = float(compute_audit["global_memory_hard_fraction"])
    if maximum >= abort_fraction or maximum >= hard_fraction:
        raise RuntimeError(
            "The continuous global CUDA guard observed memory at or above a protected limit."
        )
    return {
        "status": "PASSED",
        "memory_guard_active": True,
        "sample_count": len(history),
        "maximum_global_used_fraction": maximum,
        "abort_fraction": abort_fraction,
        "hard_fraction": hard_fraction,
        "all_samples_below_abort": True,
        "all_samples_below_hard_limit": True,
        "coverage": "pretest_model_load_through_all_locked_test_gpu_rollouts",
    }


def _load_proxy(proxy_run: Path) -> tuple[FinalProxyBundle, Path, Path]:
    manifest_path = proxy_run / "manifest.json"
    artifact_path = proxy_run / "final_proxy_bundle.joblib"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("The frozen proxy manifest is not completed.")
    bundle = joblib.load(artifact_path)
    if not isinstance(bundle, FinalProxyBundle):
        raise RuntimeError("The frozen proxy artifact has an unexpected type.")
    return bundle, manifest_path, artifact_path


def _build_formal_model(
    algorithm: str,
    env: Any,
    *,
    seed: int,
    n_envs: int,
    hidden_layers: Sequence[int],
    device: str,
    algorithm_config: Mapping[str, Any],
) -> Any:
    common = {
        "policy": "MlpPolicy",
        "env": env,
        "policy_kwargs": {"net_arch": [int(value) for value in hidden_layers]},
        "seed": int(seed),
        "device": str(device),
        "verbose": 0,
    }
    if algorithm == "PPO":
        n_steps = int(algorithm_config["n_steps_per_env"])
        batch_size = int(algorithm_config["batch_size"])
        rollout_size = n_steps * int(n_envs)
        if rollout_size % batch_size:
            raise RuntimeError("PPO batch_size must divide n_steps_per_env * n_envs.")
        return PPO(
            **common,
            learning_rate=float(algorithm_config["learning_rate"]),
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=int(algorithm_config["n_epochs"]),
            gamma=float(algorithm_config["gamma"]),
            gae_lambda=float(algorithm_config["gae_lambda"]),
            clip_range=float(algorithm_config["clip_range"]),
        )
    off_policy = {
        **common,
        "learning_rate": float(algorithm_config["learning_rate"]),
        "buffer_size": int(algorithm_config["buffer_size"]),
        "learning_starts": int(algorithm_config["learning_starts"]),
        "batch_size": int(algorithm_config["batch_size"]),
        "gamma": float(algorithm_config["gamma"]),
        "tau": float(algorithm_config["tau"]),
        "train_freq": int(algorithm_config["train_freq"]),
        "gradient_steps": int(algorithm_config["gradient_steps"]),
    }
    if algorithm == "SAC":
        return SAC(**off_policy, ent_coef=str(algorithm_config["ent_coef"]))
    if algorithm == "TD3":
        sigma = float(algorithm_config["action_noise_sigma"])
        noise = NormalActionNoise(mean=np.zeros(2), sigma=np.full(2, sigma))
        return TD3(
            **off_policy,
            action_noise=noise,
            policy_delay=int(algorithm_config["policy_delay"]),
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


class FormalValidationCallback(legacy.ValidationCheckpointCallback):
    """Add learning diagnostics to the frozen validation-only checkpoint callback."""

    def __init__(
        self,
        *,
        compute_audit: Mapping[str, Any],
        memory_watchdog: CudaMemoryWatchdog,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.training_trace: list[dict[str, Any]] = []
        self.memory_watchdog = memory_watchdog
        self._gpu_device_index = (
            int(compute_audit["device_index"])
            if compute_audit.get("device_type") == "cuda"
            else None
        )
        self._gpu_abort_fraction = float(
            compute_audit.get("global_memory_abort_fraction", 1.0)
        )
        self._gpu_check_interval = int(compute_audit.get("memory_check_interval_steps", 1))
        self._next_gpu_check = self._gpu_check_interval

    def _record_gpu_memory(self, event: str) -> dict[str, Any] | None:
        if self._gpu_device_index is None:
            return None
        row = self.memory_watchdog.sample(event, int(self.num_timesteps))
        if row is None:
            return None
        row["test_accessed"] = False
        return row

    def _evaluate(self) -> None:
        step_before = set(self.evaluated_steps)
        started = time.perf_counter()
        self._record_gpu_memory("before_validation_checkpoint")
        super()._evaluate()
        step = int(self.num_timesteps)
        if step in step_before:
            return
        episode_buffer = list(getattr(self.model, "ep_info_buffer", []) or [])
        rewards = np.asarray(
            [float(item["r"]) for item in episode_buffer if "r" in item], dtype=float
        )
        lengths = np.asarray(
            [float(item["l"]) for item in episode_buffer if "l" in item], dtype=float
        )
        logger_values = getattr(self.model.logger, "name_to_value", {})
        gpu_sample = self._record_gpu_memory("validation_checkpoint")
        row: dict[str, Any] = {
            "algorithm": self.algorithm,
            "training_seed": self.training_seed,
            "checkpoint_step": step,
            "recent_episode_reward_mean": (
                float(rewards.mean()) if rewards.size else np.nan
            ),
            "recent_episode_reward_sd": float(rewards.std(ddof=1)) if rewards.size > 1 else np.nan,
            "recent_episode_length_mean": float(lengths.mean()) if lengths.size else np.nan,
            "recent_episode_count": int(rewards.size),
            "validation_HV": float(self.history[-1]["validation_HV"]),
            "validation_support_valid_rate": float(
                self.history[-1]["validation_mean_support_valid_rate"]
            ),
            "best_validation_HV_so_far": float(self.best_hv),
            "best_checkpoint_step_so_far": int(self.best_step),
            "diagnostic_capture_seconds": time.perf_counter() - started,
            "test_accessed": False,
        }
        if gpu_sample is not None:
            row.update(
                {
                    "gpu_global_used_fraction": gpu_sample["global_used_fraction"],
                    "gpu_process_allocated_bytes": gpu_sample["process_allocated_bytes"],
                    "gpu_process_reserved_bytes": gpu_sample["process_reserved_bytes"],
                }
            )
        for key, value in logger_values.items():
            if not str(key).startswith(("train/", "rollout/", "time/")):
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                row[str(key).replace("/", "_")] = scalar
        self.training_trace.append(row)

    def _on_step(self) -> bool:
        keep_training = super()._on_step()
        if self._gpu_device_index is not None and self.num_timesteps >= self._next_gpu_check:
            self._record_gpu_memory("periodic_guard")
            while self._next_gpu_check <= self.num_timesteps:
                self._next_gpu_check += self._gpu_check_interval
        return keep_training


def _environment_payload() -> dict[str, Any]:
    payload = dict(legacy._environment())
    payload.update(
        {
            "processor": platform.processor(),
            "logical_cpu_count": __import__("os").cpu_count(),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_version": str(torch.__version__),
            "torch_compiled_cuda": torch.version.cuda,
            "torch_cuda_available": bool(torch.cuda.is_available()),
        }
    )
    return payload


def _source_hashes(config_path: Path, initial_path: Path, proxy_run: Path) -> dict[str, str]:
    paths = {
        "config": config_path,
        "initial_dataset": initial_path,
        "initial_dataset_manifest": initial_path.with_name("initial_dataset_manifest.json"),
        "proxy_manifest": proxy_run / "manifest.json",
        "proxy_artifact": proxy_run / "final_proxy_bundle.joblib",
        "formal_runner": Path(__file__),
        "legacy_runner_helpers": PROJECT_ROOT / "scripts" / "run_surrogate_rl.py",
        "surrogate_environment": PROJECT_ROOT / "src" / "taici" / "surrogate_rl.py",
        "fast_proxy": PROJECT_ROOT / "src" / "taici" / "fast_proxy.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _provenance_snapshot_hashes(output_dir: Path) -> dict[str, str]:
    provenance = output_dir / "provenance"
    paths = {
        "config": provenance / "formal_rl_rerun.toml",
        "formal_runner": provenance / "run_formal_rl_rerun.py",
        "legacy_runner_helpers": provenance / "run_surrogate_rl.py",
        "surrogate_environment": provenance / "surrogate_rl.py",
        "fast_proxy": provenance / "fast_proxy.py",
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def _development_fingerprint(
    source_and_input_hashes: Mapping[str, str],
    budget: int,
    selections: Sequence[Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        {
            "inputs": dict(source_and_input_hashes),
            "budget": int(budget),
            "selections": [dict(row) for row in selections],
        },
        sort_keys=True,
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype=np.float64))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _scenario_fit_lock(scenario: ScenarioData) -> dict[str, Any]:
    arrays = {
        "observation_mean": scenario.observation_mean,
        "observation_scale": scenario.observation_scale,
        "action_low": scenario.action_low,
        "action_high": scenario.action_high,
        "action_rate": scenario.action_rate,
        "objective_low": scenario.objective_low,
        "objective_high": scenario.objective_high,
    }
    if scenario.context_support_scaler is not None:
        arrays["context_scaler_mean"] = scenario.context_support_scaler.mean_
        arrays["context_scaler_scale"] = scenario.context_support_scaler.scale_
    if scenario.action_context_scaler is not None:
        arrays["action_context_scaler_mean"] = scenario.action_context_scaler.mean_
        arrays["action_context_scaler_scale"] = scenario.action_context_scaler.scale_
    if scenario.support_train_actions is not None:
        arrays["support_train_actions"] = scenario.support_train_actions
    return {
        "support_mode": scenario.support_mode,
        "union_feature_names": list(scenario.union_feature_names),
        "context_feature_names": list(scenario.context_feature_names),
        "action_context_feature_names": list(scenario.action_context_feature_names),
        "support_threshold_legacy": float(scenario.support_threshold),
        "context_support_threshold": (
            None
            if scenario.context_support_threshold is None
            else float(scenario.context_support_threshold)
        ),
        "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
    }


def _copy_provenance(output_dir: Path, config_path: Path) -> None:
    provenance = output_dir / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    for source in (
        config_path,
        Path(__file__),
        PROJECT_ROOT / "scripts" / "run_surrogate_rl.py",
        PROJECT_ROOT / "src" / "taici" / "surrogate_rl.py",
        PROJECT_ROOT / "src" / "taici" / "fast_proxy.py",
    ):
        shutil.copy2(source, provenance / source.name)


def _convergence_gate(
    history: pd.DataFrame,
    *,
    algorithms: Sequence[str],
    seeds: Sequence[int],
    budget_config: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    checkpoints = history.drop_duplicates(
        ["algorithm", "training_seed", "checkpoint_step"]
    )
    summary = (
        checkpoints.groupby(["algorithm", "checkpoint_step"], observed=True)
        .agg(
            mean_validation_HV=("validation_HV", "mean"),
            median_validation_HV=("validation_HV", "median"),
            sd_validation_HV=("validation_HV", "std"),
            min_validation_HV=("validation_HV", "min"),
            max_validation_HV=("validation_HV", "max"),
            mean_support_valid_rate=("validation_mean_support_valid_rate", "mean"),
            seed_count=("training_seed", "nunique"),
        )
        .reset_index()
        .sort_values(["algorithm", "checkpoint_step"])
    )
    terminal_n = int(budget_config["terminal_window_checkpoints"])
    min_checkpoints = int(budget_config["minimum_checkpoints"])
    max_gain = float(budget_config["maximum_relative_hv_gain_terminal_window"])
    max_cv = float(budget_config["maximum_seed_cv_at_selected_budget"])
    records: list[dict[str, Any]] = []
    for algorithm in algorithms:
        group = summary.loc[summary["algorithm"].eq(algorithm)].copy()
        terminal = group.tail(terminal_n)
        first_hv = float(terminal.iloc[0]["median_validation_HV"])
        last_hv = float(terminal.iloc[-1]["median_validation_HV"])
        relative_terminal_gain = (last_hv - first_hv) / max(abs(first_hv), 1e-12)
        checkpoint_rows = checkpoints.loc[checkpoints["algorithm"].eq(algorithm)]
        per_seed_best = checkpoint_rows.groupby("training_seed")["validation_HV"].max()
        seed_cv = float(per_seed_best.std(ddof=1) / abs(per_seed_best.mean()))
        record = {
            "algorithm": algorithm,
            "checkpoint_count": int(len(group)),
            "seed_count": int(per_seed_best.index.nunique()),
            "terminal_first_step": int(terminal.iloc[0]["checkpoint_step"]),
            "terminal_last_step": int(terminal.iloc[-1]["checkpoint_step"]),
            "terminal_first_median_HV": first_hv,
            "terminal_last_median_HV": last_hv,
            "relative_terminal_HV_gain": relative_terminal_gain,
            "absolute_relative_terminal_HV_gain": abs(relative_terminal_gain),
            "best_HV_seed_CV": seed_cv,
            "checkpoint_count_pass": bool(len(group) >= min_checkpoints),
            "seed_count_pass": bool(per_seed_best.index.nunique() == len(seeds)),
            "terminal_plateau_pass": bool(abs(relative_terminal_gain) <= max_gain),
            "seed_stability_pass": bool(seed_cv <= max_cv),
        }
        record["algorithm_gate_pass"] = bool(
            record["checkpoint_count_pass"]
            and record["seed_count_pass"]
            and record["terminal_plateau_pass"]
            and record["seed_stability_pass"]
        )
        records.append(record)
    gate_table = pd.DataFrame.from_records(records)
    report = {
        "status": "PASSED" if gate_table["algorithm_gate_pass"].all() else "EXTEND",
        "all_algorithms_pass": bool(gate_table["algorithm_gate_pass"].all()),
        "test_authorized": False,
        "test_remains_sealed": True,
        "rule": str(budget_config["rule"]),
        "thresholds": dict(budget_config),
        "algorithms": records,
    }
    return summary, report


def _development(args: argparse.Namespace) -> int:
    config_path = _resolve(args.config)
    config = load_toml(str(config_path))
    rl_config = dict(config["rl"])
    compute_audit = _configure_compute(rl_config)
    registered = {
        int(rl_config["development_max_timesteps_per_algorithm_seed"]),
        int(rl_config["extension_max_timesteps_per_algorithm_seed"]),
    }
    total_timesteps = int(
        args.timesteps
        if args.timesteps is not None
        else (1000 if args.smoke else rl_config["development_max_timesteps_per_algorithm_seed"])
    )
    if not args.smoke and total_timesteps not in registered:
        raise RuntimeError(f"Development timesteps must be one of {sorted(registered)}.")
    algorithms = tuple(str(value) for value in rl_config["algorithms"])
    seeds = tuple(int(value) for value in rl_config["training_seeds"])
    preferences = tuple(float(value) for value in rl_config["preference_weights"])
    if algorithms != FORMAL_ALGORITHMS or seeds != FORMAL_TRAINING_SEEDS:
        raise RuntimeError("The formal matrix must contain PPO/SAC/TD3 and five unique seeds.")
    if preferences != (0.25, 0.50, 0.75):
        raise RuntimeError("The registered preferences are 0.25, 0.50 and 0.75.")
    if args.smoke:
        algorithms, seeds = algorithms[:1], seeds[:1]
    checkpoint_frequency = (
        total_timesteps if args.smoke else int(rl_config["checkpoint_frequency"])
    )
    if total_timesteps % checkpoint_frequency:
        raise RuntimeError("The common budget must be divisible by checkpoint_frequency.")

    initial_path = _resolve(str(config["inputs"]["initial_dataset"]))
    proxy_run = _resolve(str(config["inputs"]["proxy_run"]))
    startup_hashes = _source_hashes(config_path, initial_path, proxy_run)
    gate_path = _resolve(str(config["inputs"]["formal_plant_gate_status"]))
    plant_gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if (
        plant_gate.get("status") != "BLOCKED"
        or bool(plant_gate.get("phase9_authorized"))
        or int(plant_gate.get("factory_optimizer_calls", -1)) != 0
    ):
        raise RuntimeError("The independent proxy experiment must preserve the blocked plant gate.")

    timestamp = _utc_now()
    if args.output_dir:
        output_dir = _resolve(args.output_dir)
    else:
        output_root = _resolve(str(config["output"]["root"]))
        prefix = "smoke_development" if args.smoke else "development"
        output_dir = output_root / f"{prefix}_{total_timesteps}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    _copy_provenance(output_dir, config_path)
    provenance_hashes = _provenance_snapshot_hashes(output_dir)
    for name, digest in provenance_hashes.items():
        if digest != startup_hashes[name]:
            raise RuntimeError(f"Startup provenance snapshot mismatch: {name}.")
    source_lock = {
        "status": "LOCKED_AT_DEVELOPMENT_START",
        "created_utc": timestamp,
        "active_source_and_input_hashes": startup_hashes,
        "provenance_snapshot_hashes": provenance_hashes,
        "test_accessed": False,
    }
    write_json(source_lock, output_dir / "source_input_lock_at_start.json")
    status_path = output_dir / "run_status.json"
    running_status = (
        "SMOKE_RUNNING_TEST_SEALED_NOT_AUTHORIZABLE"
        if args.smoke
        else "DEVELOPMENT_RUNNING_TEST_SEALED"
    )
    write_json(
        {
            "status": running_status,
            "created_utc": timestamp,
            "test_accessed": False,
            "total_timesteps_per_algorithm_seed": total_timesteps,
        },
        status_path,
    )
    write_json(compute_audit, output_dir / "compute_device_audit.json")
    started = time.perf_counter()
    try:
        initial = load_initial_dataset(initial_path)
        development_cutoff = pd.Timestamp(str(rl_config["validation_end"]))
        development_frame = initial.frame.loc[
            pd.to_datetime(initial.frame["Date"]) <= development_cutoff
        ].copy().reset_index(drop=True)
        development_frame.to_csv(
            output_dir / "provenance" / "development_input_snapshot.csv",
            index=False,
            encoding=str(config["output"]["encoding"]),
        )
        development_feature_set = _as_feature_set(development_frame)
        original_bundle, proxy_manifest_path, proxy_artifact_path = _load_proxy(proxy_run)
        thread_changes = legacy._limit_estimator_threads(original_bundle)
        thread_audit = {
            "status": "PASSED",
            "scope": "inference_thread_parameters_only_no_test_panel_prediction",
            "changes": thread_changes,
            "test_policy_rollouts_accessed": False,
        }
        fast_bundle = FastExactProxyBundle.from_frozen(original_bundle)
        tn_matrix = development_feature_set.bundles["TN_out"].matrix(FEATURE_KEY)
        dec_matrix = development_feature_set.bundles["DEC"].matrix(FEATURE_KEY)
        development_proxy_audit = audit_fast_proxy_fidelity(
            original_bundle,
            fast_bundle,
            tn_matrix,
            dec_matrix,
            tolerance=1e-9,
        )
        development_rl_config = dict(rl_config)
        development_rl_config["test_start"] = rl_config["validation_start"]
        development_rl_config["test_end"] = rl_config["validation_end"]
        scenario = build_scenario_data(
            development_feature_set,
            development_frame,
            original_bundle,
            development_rl_config,
        )
        write_json(thread_audit, output_dir / "inference_thread_fidelity_audit.json")
        write_json(
            {
                "status": "PASSED",
                "audit_scope": "training_and_validation_dates_only",
                "test_policy_rollouts_accessed": False,
                "development_panel": development_proxy_audit,
                "fast_proxy_is_numerically_equivalent_not_distilled": True,
            },
            output_dir / "development_fast_proxy_fidelity_audit.json",
        )
        write_json(_environment_payload(), output_dir / "environment.json")
        scenario_contract = {
            "analysis_role": "historical_support_domain_constrained_surrogate_simulation",
            "development_input_end": development_cutoff.date().isoformat(),
            "training_period": [rl_config["train_start"], rl_config["train_end"]],
            "validation_period": [
                rl_config["validation_start"],
                rl_config["validation_end"],
            ],
            "locked_retrospective_panel_period": [
                rl_config["test_start"],
                rl_config["test_end"],
            ],
            "train_episode_starts": len(scenario.train_starts),
            "validation_episode_starts": len(scenario.validation_starts),
            "episode_horizon_days": scenario.episode_horizon,
            "action_variables": ["PPA", "DO"],
            "slow_state": "MLSS",
            "support_mode": scenario.support_mode,
            "context_feature_names": list(scenario.context_feature_names),
            "conditional_action_context_feature_names": list(
                scenario.action_context_feature_names
            ),
            "test_role": "locked_retrospective_policy_evaluation_not_independent_test",
            "upstream_proxy_independent_test": False,
            "test_policy_rollouts_accessed": False,
        }
        write_json(scenario_contract, output_dir / "scenario_contract.json")
        audit_env = legacy._make_env(
            scenario, fast_bundle, development_rl_config, scenario.train_starts
        )
        legacy.check_env(audit_env, warn=True, skip_render_check=True)
        audit_env.close()
        write_json(
            {
                "status": "PASSED",
                "gymnasium_check_env": True,
                "development_only": True,
                "test_policy_rollouts_accessed": False,
            },
            output_dir / "environment_audit.json",
        )
        write_json(_scenario_fit_lock(scenario), output_dir / "scenario_fit_lock.json")

        benchmark_env = legacy._make_env(
            scenario, fast_bundle, rl_config, scenario.train_starts
        )
        speed = benchmark_proxy_step(benchmark_env, steps=30)
        benchmark_env.close()
        write_json(
            {
                "single_environment": speed,
                "historical_8000_step_wall_seconds_range": [279.773553, 386.084476],
                "budget_selection_role": "pre_test_compute_feasibility",
                "test_accessed": False,
            },
            output_dir / "pretraining_speed_evidence.json",
        )

        validation_history: list[dict[str, Any]] = []
        training_trace: list[dict[str, Any]] = []
        gpu_memory_history: list[dict[str, Any]] = []
        runtime_records: list[dict[str, Any]] = []
        best_models: list[dict[str, Any]] = []
        encoding = str(config["output"]["encoding"])
        n_envs = int(rl_config["training_parallel_envs"])
        validation_starts = (
            scenario.validation_starts[:2] if args.smoke else scenario.validation_starts
        )
        for algorithm in algorithms:
            for seed in seeds:
                print(
                    f"[formal-rl/develop] {algorithm} seed={seed} "
                    f"budget={total_timesteps}; test sealed",
                    flush=True,
                )
                model_dir = models_dir / algorithm / f"seed_{seed}"
                model_dir.mkdir(parents=True, exist_ok=False)
                train_env = legacy._make_training_env(
                    scenario, fast_bundle, rl_config, seed=seed, n_envs=n_envs
                )
                validation_env = legacy._make_env(
                    scenario, fast_bundle, rl_config, validation_starts
                )
                model_watchdog = CudaMemoryWatchdog(
                    compute_audit, phase="development_training"
                )
                model_watchdog.set_context(algorithm=algorithm, training_seed=seed)
                model: Any | None = None
                try:
                    model_watchdog.start()
                    model_watchdog.sample("before_model_build", 0)
                    model = _build_formal_model(
                        algorithm,
                        train_env,
                        seed=seed,
                        n_envs=n_envs,
                        hidden_layers=tuple(rl_config["policy_hidden_layers"]),
                        device=str(rl_config["device"]),
                        algorithm_config=config["algorithms"][algorithm],
                    )
                    actual_model_device = _assert_model_device(
                        model, str(rl_config["device"])
                    )
                    model_watchdog.sample("after_model_build", 0)
                    callback = FormalValidationCallback(
                        algorithm=algorithm,
                        training_seed=seed,
                        validation_env=validation_env,
                        validation_starts=validation_starts,
                        preferences=preferences,
                        evaluation_frequency=checkpoint_frequency,
                        model_dir=model_dir,
                        objective_low=scenario.objective_low,
                        objective_high=scenario.objective_high,
                        reference_point=tuple(rl_config["hypervolume_reference"]),
                        compute_audit=compute_audit,
                        memory_watchdog=model_watchdog,
                    )
                    train_started = time.perf_counter()
                    model_watchdog.sample("before_model_learn", 0)
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        progress_bar=False,
                    )
                    model_watchdog.sample("after_model_learn", int(model.num_timesteps))
                    model_watchdog.assert_safe()
                    elapsed = time.perf_counter() - train_started
                    model_watchdog.sample("before_model_save", int(model.num_timesteps))
                    model.save(model_dir / "final_model.zip")
                    model_watchdog.sample("after_model_save", int(model.num_timesteps))
                    if callback.best_step < 0 or not callback.best_model_path.exists():
                        raise RuntimeError(f"No best validation model for {algorithm}/{seed}.")
                    validation_history.extend(callback.history)
                    training_trace.extend(callback.training_trace)
                    pd.DataFrame.from_records(validation_history).to_csv(
                        output_dir / "validation_history.csv", index=False, encoding=encoding
                    )
                    pd.DataFrame.from_records(training_trace).to_csv(
                        output_dir / "training_curve.csv", index=False, encoding=encoding
                    )
                    max_gpu_used = (
                        max(
                            float(row["global_used_fraction"])
                            for row in model_watchdog.history
                        )
                        if model_watchdog.history
                        else np.nan
                    )
                    runtime_records.append(
                        {
                            "phase": "training_plus_validation",
                            "algorithm": algorithm,
                            "training_seed": seed,
                            "configured_timesteps": total_timesteps,
                            "actual_timesteps": int(model.num_timesteps),
                            "wall_seconds": elapsed,
                            "best_validation_step": int(callback.best_step),
                            "best_validation_HV": float(callback.best_hv),
                            "best_validation_support_valid_rate": float(
                                callback.best_support_valid_rate
                            ),
                            "compute_device": str(rl_config["device"]),
                            "actual_model_device": actual_model_device,
                            "max_gpu_global_used_fraction": max_gpu_used,
                            "test_accessed": False,
                        }
                    )
                    best_models.append(
                        {
                            "algorithm": algorithm,
                            "training_seed": seed,
                            "model_path": callback.best_model_path.relative_to(output_dir).as_posix(),
                            "model_sha256": sha256_file(callback.best_model_path),
                            "best_validation_step": int(callback.best_step),
                            "best_validation_HV": float(callback.best_hv),
                            "best_validation_support_valid_rate": float(
                                callback.best_support_valid_rate
                            ),
                            "actual_model_device": actual_model_device,
                        }
                    )
                finally:
                    model_watchdog.stop()
                    cleanup_errors: list[BaseException] = []
                    if model is not None:
                        del model
                        model = None
                    if compute_audit.get("device_type") == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except BaseException as cleanup_exc:  # pragma: no cover - defensive path
                            cleanup_errors.append(cleanup_exc)
                    gc.collect()
                    for environment in (train_env, validation_env):
                        try:
                            environment.close()
                        except BaseException as cleanup_exc:  # pragma: no cover - defensive path
                            cleanup_errors.append(cleanup_exc)
                    gpu_memory_history.extend(model_watchdog.history)
                    if gpu_memory_history:
                        pd.DataFrame.from_records(gpu_memory_history).to_csv(
                            output_dir / "gpu_memory_monitor.csv",
                            index=False,
                            encoding=encoding,
                        )
                    if cleanup_errors:
                        raise RuntimeError(
                            "Development model cleanup failed: "
                            + "; ".join(
                                f"{type(item).__name__}: {item}"
                                for item in cleanup_errors
                            )
                        ) from cleanup_errors[0]
                model_watchdog.assert_safe()

        gpu_guard = _gpu_guard_summary(
            gpu_memory_history,
            compute_audit,
            expected_pairs=[
                (algorithm, seed) for algorithm in algorithms for seed in seeds
            ],
        )
        write_json(gpu_guard, output_dir / "gpu_memory_guard_summary.json")
        history_frame = pd.DataFrame.from_records(validation_history)
        if args.smoke:
            curve_summary = history_frame.drop_duplicates(
                ["algorithm", "training_seed", "checkpoint_step"]
            )
            budget_report = {
                "status": "SMOKE_ONLY",
                "all_algorithms_pass": False,
                "test_authorized": False,
                "test_remains_sealed": True,
            }
        else:
            curve_summary, budget_report = _convergence_gate(
                history_frame,
                algorithms=algorithms,
                seeds=seeds,
                budget_config=config["budget_gate"],
            )
        curve_summary.to_csv(
            output_dir / "validation_convergence_summary.csv",
            index=False,
            encoding=encoding,
        )
        pd.DataFrame.from_records(runtime_records).to_csv(
            output_dir / "runtime_summary.csv", index=False, encoding=encoding
        )
        write_json(best_models, output_dir / "frozen_validation_selections.json")
        write_json(budget_report, output_dir / "budget_gate_report.json")
        end_hashes = _source_hashes(config_path, initial_path, proxy_run)
        if end_hashes != startup_hashes:
            raise RuntimeError(
                "A registered source, configuration, or input changed during development."
            )
        hashes = dict(startup_hashes)
        fingerprint = _development_fingerprint(hashes, total_timesteps, best_models)
        manifest = {
            "stage": "formal_rl_development_smoke" if args.smoke else "formal_rl_development",
            "status": "completed_test_sealed",
            "created_utc": timestamp,
            "completed_utc": _utc_now(),
            "fingerprint": fingerprint,
            "elapsed_seconds": time.perf_counter() - started,
            "algorithms": list(algorithms),
            "training_seeds": list(seeds),
            "preference_weights_TN": list(preferences),
            "common_interaction_budget_per_algorithm_seed": total_timesteps,
            "policy_models_completed": len(best_models),
            "validation_only_checkpoint_selection": True,
            "test_policy_rollouts_accessed": False,
            "upstream_proxy_independent_test": False,
            "proxy_simulation_only": True,
            "plant_control_claim_authorized": False,
            "source_and_input_hashes": hashes,
            "source_input_lock_at_start_sha256": sha256_file(
                output_dir / "source_input_lock_at_start.json"
            ),
            "proxy_manifest": str(proxy_manifest_path),
            "proxy_artifact": str(proxy_artifact_path),
            "budget_gate_status": budget_report["status"],
            "compute_device": str(rl_config["device"]),
            "compute_device_audit_sha256": sha256_file(
                output_dir / "compute_device_audit.json"
            ),
            "gpu_memory_monitor_sha256": (
                sha256_file(output_dir / "gpu_memory_monitor.csv")
                if (output_dir / "gpu_memory_monitor.csv").exists()
                else None
            ),
            "gpu_memory_guard_summary_sha256": sha256_file(
                output_dir / "gpu_memory_guard_summary.json"
            ),
            "gpu_memory_guard": gpu_guard,
        }
        write_json(manifest, output_dir / "development_manifest.json")
        completion_status = (
            "SMOKE_COMPLETE_TEST_SEALED_NOT_AUTHORIZABLE"
            if args.smoke
            else "DEVELOPMENT_COMPLETE_TEST_SEALED"
        )
        write_json(
            {
                "status": completion_status,
                "created_utc": timestamp,
                "completed_utc": _utc_now(),
                "fingerprint": fingerprint,
                "budget_gate_status": budget_report["status"],
                "test_accessed": False,
                "test_authorized": False,
            },
            status_path,
        )
        print(
            json.dumps(
                {
                    "status": (
                        "smoke_complete_test_sealed_not_authorizable"
                        if args.smoke
                        else "development_complete_test_sealed"
                    ),
                    "run_dir": str(output_dir),
                    "budget_gate": budget_report["status"],
                    "elapsed_seconds": manifest["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        if "gpu_memory_history" in locals() and gpu_memory_history:
            pd.DataFrame.from_records(gpu_memory_history).to_csv(
                output_dir / "gpu_memory_monitor.csv",
                index=False,
                encoding=str(config["output"]["encoding"]),
            )
            unsafe_rows = [
                row
                for row in gpu_memory_history
                if float(row["global_used_fraction"])
                >= float(compute_audit.get("global_memory_abort_fraction", 1.0))
            ]
            if unsafe_rows:
                write_json(
                    {
                        "status": "GPU_MEMORY_GUARD_TRIGGERED",
                        "threshold": compute_audit.get(
                            "global_memory_abort_fraction"
                        ),
                        "first_unsafe_sample": unsafe_rows[0],
                    },
                    output_dir / "gpu_memory_breach.json",
                )
        write_json(
            {
                "status": (
                    "SMOKE_FAILED_TEST_SEALED_NOT_AUTHORIZABLE"
                    if args.smoke
                    else "DEVELOPMENT_FAILED_TEST_SEALED"
                ),
                "failed_utc": _utc_now(),
                "test_accessed": False,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_path,
        )
        raise


def _seedwise_metrics(
    points: pd.DataFrame,
    union_front: pd.DataFrame,
    reference_point: Sequence[float],
) -> pd.DataFrame:
    reference = union_front[["normalized_TN", "normalized_DEC"]].to_numpy(float)
    ref_point = np.asarray(reference_point, dtype=float)
    hv_indicator = HV(ref_point=ref_point)
    records: list[dict[str, Any]] = []
    for (method, role, seed), group in points.groupby(
        ["method", "method_role", "training_seed"], observed=True
    ):
        values = group[["normalized_TN", "normalized_DEC"]].to_numpy(float)
        front = values[non_dominated_mask(values)]
        inside = np.all(front < ref_point, axis=1)
        records.append(
            {
                "method": method,
                "method_role": role,
                "seed": int(seed),
                "hypervolume": float(hv_indicator(front[inside])) if inside.any() else 0.0,
                "IGD_plus_to_empirical_union_front": float(IGDPlus(reference)(front)),
                "non_dominated_points": int(len(front)),
                "mean_support_valid_rate": float(group["support_valid_rate"].mean()),
                "mean_repair_rate": float(group["repair_rate"].mean()),
                "total_preference_points": int(len(group)),
            }
        )
    return pd.DataFrame.from_records(records).sort_values(
        ["method", "seed"]
    ).reset_index(drop=True)


def _failure_cases(episode_summary: pd.DataFrame) -> pd.DataFrame:
    rl = episode_summary.loc[episode_summary["method_role"].eq("learned_RL_policy")].copy()
    random = episode_summary.loc[episode_summary["method"].eq("RandomFeasible")].copy()
    keys = ["training_seed", "preference_TN", "episode_start_position"]
    random = random.loc[:, keys + ["mean_TN_out", "mean_DEC"]].rename(
        columns={"mean_TN_out": "random_TN_out", "mean_DEC": "random_DEC"}
    )
    paired = rl.merge(random, on=keys, how="left", validate="many_to_one")
    if paired[["random_TN_out", "random_DEC"]].isna().any().any():
        raise RuntimeError("RandomFeasible lacks a matched seed/preference/test episode.")
    paired["delta_TN_vs_RandomFeasible"] = paired["mean_TN_out"] - paired["random_TN_out"]
    paired["delta_DEC_vs_RandomFeasible"] = paired["mean_DEC"] - paired["random_DEC"]
    paired["dominated_by_RandomFeasible"] = (
        (paired["delta_TN_vs_RandomFeasible"] >= 0)
        & (paired["delta_DEC_vs_RandomFeasible"] >= 0)
    )
    return paired.sort_values(
        ["dominated_by_RandomFeasible", "delta_TN_vs_RandomFeasible", "delta_DEC_vs_RandomFeasible"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _validate_locked_test_contract(
    run_dir: Path,
    status: Mapping[str, Any],
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    Path,
    Path,
    list[dict[str, Any]],
]:
    if status.get("status") != "DEVELOPMENT_COMPLETE_TEST_SEALED":
        raise RuntimeError("The development run is not a formal completed sealed run.")
    if status.get("test_accessed") is not False or status.get("test_authorized") is not False:
        raise RuntimeError("The development status no longer represents an unopened test.")
    if manifest.get("stage") != "formal_rl_development":
        raise RuntimeError("Smoke or non-formal development runs cannot open the test panel.")
    if manifest.get("status") != "completed_test_sealed":
        raise RuntimeError("The development manifest is not completed and sealed.")
    if tuple(manifest.get("algorithms", ())) != FORMAL_ALGORITHMS:
        raise RuntimeError("The development manifest does not contain all formal algorithms.")
    manifest_seeds = tuple(int(value) for value in manifest.get("training_seeds", ()))
    if manifest_seeds != FORMAL_TRAINING_SEEDS:
        raise RuntimeError("The development manifest does not contain all formal seeds.")
    budget = int(manifest.get("common_interaction_budget_per_algorithm_seed", -1))
    if budget != FORMAL_INTERACTION_BUDGET or int(manifest.get("policy_models_completed", -1)) != 15:
        raise RuntimeError("One-shot testing requires the complete 15-model 32k development run.")
    fingerprint = str(manifest.get("fingerprint", ""))
    if status.get("fingerprint") != fingerprint:
        raise RuntimeError("run_status and development_manifest fingerprints differ.")

    if authorization.get("schema_version") != "formal_rl_budget_freeze_v1":
        raise RuntimeError("The budget-freeze schema is not recognized.")
    if authorization.get("status") != "FORMAL_DEVELOPMENT_FROZEN_TEST_AUTHORIZED":
        raise RuntimeError("The formal development freeze has not authorized testing.")
    if authorization.get("test_authorized") is not True:
        raise RuntimeError("The budget freeze does not authorize the one-shot test.")
    if authorization.get("repeat_test_access_authorized") is not False:
        raise RuntimeError("Repeat test access must remain prohibited.")
    if authorization.get("test_accessed_at_freeze") is not False:
        raise RuntimeError("The budget freeze was not created before test access.")
    if int(authorization.get("selected_budget", -1)) != FORMAL_INTERACTION_BUDGET:
        raise RuntimeError("The budget freeze does not select the registered 32k budget.")
    if authorization.get("development_fingerprint") != fingerprint:
        raise RuntimeError("The budget freeze authorizes a different development fingerprint.")

    config_path = run_dir / "provenance" / "formal_rl_rerun.toml"
    config = load_toml(str(config_path))
    rl_config = dict(config["rl"])
    if tuple(str(value) for value in rl_config["algorithms"]) != FORMAL_ALGORITHMS:
        raise RuntimeError("Frozen configuration algorithms differ from the formal registry.")
    if tuple(int(value) for value in rl_config["training_seeds"]) != FORMAL_TRAINING_SEEDS:
        raise RuntimeError("Frozen configuration seeds differ from the formal registry.")
    if int(rl_config["extension_max_timesteps_per_algorithm_seed"]) != FORMAL_INTERACTION_BUDGET:
        raise RuntimeError("Frozen configuration maximum budget differs from the registry.")
    if str(rl_config["device"]).strip().lower() != "cuda:0":
        raise RuntimeError("The frozen formal test configuration must use cuda:0.")
    initial_path = _resolve(str(config["inputs"]["initial_dataset"]))
    proxy_run = _resolve(str(config["inputs"]["proxy_run"]))

    expected_hashes = dict(manifest["source_and_input_hashes"])
    current_hashes = _source_hashes(config_path, initial_path, proxy_run)
    if current_hashes != expected_hashes:
        raise RuntimeError("Frozen source/input hashes changed before test access.")
    source_lock_path = run_dir / "source_input_lock_at_start.json"
    if sha256_file(source_lock_path) != manifest.get("source_input_lock_at_start_sha256"):
        raise RuntimeError("The development-start source lock hash changed.")
    source_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
    if source_lock.get("active_source_and_input_hashes") != expected_hashes:
        raise RuntimeError("The development-start active hashes differ from the manifest.")
    snapshot_hashes = _provenance_snapshot_hashes(run_dir)
    if source_lock.get("provenance_snapshot_hashes") != snapshot_hashes:
        raise RuntimeError("A provenance source snapshot changed before test access.")
    for name, digest in snapshot_hashes.items():
        if digest != expected_hashes[name]:
            raise RuntimeError(f"Provenance snapshot differs from the startup lock: {name}.")

    selections_path = run_dir / "frozen_validation_selections.json"
    selections_raw = json.loads(selections_path.read_text(encoding="utf-8"))
    if not isinstance(selections_raw, list) or len(selections_raw) != 15:
        raise RuntimeError("Frozen selections must contain exactly 15 records.")
    selections = [dict(row) for row in selections_raw]
    if _development_fingerprint(expected_hashes, budget, selections) != fingerprint:
        raise RuntimeError("Frozen selections no longer reproduce the development fingerprint.")

    expected_pairs = {
        (algorithm, seed)
        for algorithm in FORMAL_ALGORITHMS
        for seed in FORMAL_TRAINING_SEEDS
    }
    observed_pairs: set[tuple[str, int]] = set()
    models_root = (run_dir / "models").resolve()
    for selection in selections:
        algorithm = str(selection.get("algorithm", ""))
        seed = int(selection.get("training_seed", -1))
        pair = (algorithm, seed)
        if pair not in expected_pairs or pair in observed_pairs:
            raise RuntimeError(f"Invalid or duplicate frozen selection: {pair}.")
        observed_pairs.add(pair)
        expected_relative = Path("models") / algorithm / f"seed_{seed}" / "best_model.zip"
        relative = Path(str(selection.get("model_path", "")))
        if relative != expected_relative or relative.is_absolute():
            raise RuntimeError(f"Frozen model path differs from the registered layout: {pair}.")
        model_path = (run_dir / relative).resolve()
        if models_root not in model_path.parents or not model_path.is_file():
            raise RuntimeError(f"Frozen model is outside the run or missing: {pair}.")
        registered_hash = str(selection.get("model_sha256", ""))
        if len(registered_hash) != 64 or sha256_file(model_path) != registered_hash:
            raise RuntimeError(f"Frozen model hash changed: {pair}.")
    if observed_pairs != expected_pairs:
        raise RuntimeError("Frozen selections do not cover the complete formal matrix.")

    authorized_models = {
        (str(row.get("algorithm", "")), int(row.get("training_seed", -1))): str(
            row.get("model_sha256", "")
        )
        for row in authorization.get("frozen_model_hashes", [])
        if isinstance(row, dict)
    }
    selected_models = {
        (str(row["algorithm"]), int(row["training_seed"])): str(row["model_sha256"])
        for row in selections
    }
    if authorized_models != selected_models:
        raise RuntimeError("Budget-freeze model hashes differ from frozen selections.")
    return config, rl_config, initial_path, proxy_run, selections


def _preflight_frozen_models(
    run_dir: Path,
    selections: Sequence[Mapping[str, Any]],
    rl_config: Mapping[str, Any],
    compute_audit: Mapping[str, Any],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[tuple[str, int], Any],
    CudaMemoryWatchdog,
]:
    records: list[dict[str, Any]] = []
    memory_history: list[dict[str, Any]] = []
    loaded_models: dict[tuple[str, int], Any] = {}
    global_watchdog = CudaMemoryWatchdog(
        compute_audit, phase="pretest_through_locked_test_global"
    )
    global_watchdog.set_context(algorithm="GLOBAL", training_seed=0)
    try:
        global_watchdog.start()
        for selection in selections:
            algorithm = str(selection["algorithm"])
            seed = int(selection["training_seed"])
            model_path = run_dir / str(selection["model_path"])
            watchdog = CudaMemoryWatchdog(compute_audit, phase="pretest_model_load")
            watchdog.set_context(algorithm=algorithm, training_seed=seed)
            model: Any | None = None
            try:
                watchdog.start()
                watchdog.sample("before_pretest_model_load")
                model = ALGORITHM_CLASSES[algorithm].load(
                    model_path, device=str(rl_config["device"])
                )
                actual_device = _assert_model_device(model, str(rl_config["device"]))
                watchdog.sample("after_pretest_model_load")
            except BaseException:
                if model is not None:
                    del model
                    model = None
                if compute_audit.get("device_type") == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                raise
            finally:
                watchdog.stop()
                memory_history.extend(watchdog.history)
            try:
                watchdog.assert_safe()
            except BaseException:
                if model is not None:
                    del model
                if compute_audit.get("device_type") == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                raise
            loaded_models[(algorithm, seed)] = model
            model = None
            records.append(
                {
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "model_path": str(selection["model_path"]),
                    "model_sha256": str(selection["model_sha256"]),
                    "actual_model_device": actual_device,
                    "pretest_panel_accessed": False,
                }
            )
        guard = _gpu_guard_summary(
            memory_history,
            compute_audit,
            expected_pairs=[
                (str(row["algorithm"]), int(row["training_seed"]))
                for row in selections
            ],
        )
    except BaseException:
        global_watchdog.stop()
        loaded_models.clear()
        if compute_audit.get("device_type") == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise
    return records, memory_history, guard, loaded_models, global_watchdog


def _test_once(args: argparse.Namespace) -> int:
    if not args.run_dir:
        raise ValueError("--run-dir is required for the one-shot test phase.")
    run_dir = _resolve(args.run_dir)
    status_path = run_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    authorization_path = run_dir / "budget_freeze.json"
    if not authorization_path.exists():
        raise RuntimeError("budget_freeze.json is required before the one-shot test.")
    authorization = json.loads(authorization_path.read_text(encoding="utf-8"))
    manifest = json.loads(
        (run_dir / "development_manifest.json").read_text(encoding="utf-8")
    )
    config, rl_config, initial_path, proxy_run, selections = (
        _validate_locked_test_contract(run_dir, status, authorization, manifest)
    )
    compute_audit = _configure_compute(rl_config)
    test_dir = run_dir / "test_once"
    if test_dir.exists():
        raise RuntimeError("The one-shot test directory already exists; repeat access is forbidden.")

    (
        pretest_records,
        pretest_gpu_history,
        pretest_gpu_guard,
        preloaded_models,
        test_global_watchdog,
    ) = _preflight_frozen_models(run_dir, selections, rl_config, compute_audit)
    try:
        initial = load_initial_dataset(initial_path)
        original_bundle, _, _ = _load_proxy(proxy_run)
        legacy._limit_estimator_threads(original_bundle)
        fast_bundle = FastExactProxyBundle.from_frozen(original_bundle)
        scenario = build_scenario_data(
            initial.feature_set, initial.frame, original_bundle, rl_config
        )
        frozen_fit = json.loads(
            (run_dir / "scenario_fit_lock.json").read_text(encoding="utf-8")
        )
        current_fit = _scenario_fit_lock(scenario)
        if current_fit != frozen_fit:
            raise RuntimeError(
                "Training-fitted environment parameters changed before retrospective evaluation."
            )
    except BaseException:
        test_global_watchdog.stop()
        preloaded_models.clear()
        if compute_audit.get("device_type") == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        raise

    access_path = test_dir / "TEST_ACCESS.json"
    access_started = _utc_now()
    test_access_committed = False
    started = 0.0
    try:
        test_dir.mkdir(parents=False, exist_ok=False)
        write_json(
            {
                "status": "TEST_ACCESS_STARTED",
                "started_utc": access_started,
                "development_fingerprint": manifest["fingerprint"],
                "test_accessed": True,
                "test_access_attempts": 1,
                "repeat_access_allowed": False,
            },
            access_path,
        )
        test_access_committed = True
        write_json(
            {
                "status": "FORMAL_RL_TEST_ACCESS_IN_PROGRESS",
                "development_fingerprint": manifest["fingerprint"],
                "test_accessed": True,
                "test_access_attempts": 1,
                "test_started_utc": access_started,
            },
            status_path,
        )
        write_json(compute_audit, test_dir / "compute_device_audit.json")
        write_json(pretest_records, test_dir / "pretest_model_load_audit.json")
        write_json(pretest_gpu_guard, test_dir / "pretest_gpu_memory_guard_summary.json")
        pd.DataFrame.from_records(pretest_gpu_history).to_csv(
            test_dir / "pretest_gpu_memory_monitor.csv",
            index=False,
            encoding=str(config["output"]["encoding"]),
        )
        started = time.perf_counter()
        dates = pd.DatetimeIndex(
            pd.to_datetime(initial.feature_set.bundles["TN_out"].anchor["Date"])
        )
        test_mask = (dates >= pd.Timestamp(str(rl_config["test_start"]))) & (
            dates <= pd.Timestamp(str(rl_config["test_end"]))
        )
        test_fidelity = audit_fast_proxy_fidelity(
            original_bundle,
            fast_bundle,
            initial.feature_set.bundles["TN_out"].matrix(FEATURE_KEY).loc[test_mask],
            initial.feature_set.bundles["DEC"].matrix(FEATURE_KEY).loc[test_mask],
            tolerance=1e-9,
        )
        write_json(test_fidelity, test_dir / "test_fast_proxy_fidelity_audit.json")

        algorithms = tuple(str(value) for value in rl_config["algorithms"])
        preferences = tuple(float(value) for value in rl_config["preference_weights"])
        trajectories: list[pd.DataFrame] = []
        runtime: list[dict[str, Any]] = []
        test_gpu_memory_history: list[dict[str, Any]] = []
        encoding = str(config["output"]["encoding"])
        for selection in selections:
            algorithm = str(selection["algorithm"])
            seed = int(selection["training_seed"])
            model_watchdog = CudaMemoryWatchdog(
                compute_audit, phase="locked_test_policy_rollout"
            )
            model_watchdog.set_context(algorithm=algorithm, training_seed=seed)
            model = preloaded_models.pop((algorithm, seed))
            env: Any | None = None
            try:
                model_watchdog.start()
                actual_model_device = _assert_model_device(
                    model, str(rl_config["device"])
                )
                env = legacy._make_env(
                    scenario, fast_bundle, rl_config, scenario.test_starts
                )
                phase_started = time.perf_counter()
                model_watchdog.sample("before_policy_rollout")
                trajectories.append(
                    rollout_policy(
                        model,
                        env,
                        algorithm=algorithm,
                        training_seed=seed,
                        start_positions=scenario.test_starts,
                        preferences=preferences,
                    )
                )
                model_watchdog.sample("after_policy_rollout")
                model_watchdog.assert_safe()
                runtime.append(
                    {
                        "phase": "locked_test_policy_rollout",
                        "method": algorithm,
                        "seed": seed,
                        "wall_seconds": time.perf_counter() - phase_started,
                        "compute_device": str(rl_config["device"]),
                        "actual_model_device": actual_model_device,
                        "max_gpu_global_used_fraction": (
                            max(
                                float(row["global_used_fraction"])
                                for row in model_watchdog.history
                            )
                            if model_watchdog.history
                            else np.nan
                        ),
                    }
                )
            finally:
                model_watchdog.stop()
                cleanup_errors: list[BaseException] = []
                if env is not None:
                    try:
                        env.close()
                    except BaseException as cleanup_exc:  # pragma: no cover - defensive path
                        cleanup_errors.append(cleanup_exc)
                if model is not None:
                    del model
                    model = None
                if compute_audit.get("device_type") == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except BaseException as cleanup_exc:  # pragma: no cover - defensive path
                        cleanup_errors.append(cleanup_exc)
                gc.collect()
                test_gpu_memory_history.extend(model_watchdog.history)
                if test_gpu_memory_history:
                    pd.DataFrame.from_records(test_gpu_memory_history).to_csv(
                        test_dir / "gpu_memory_monitor.csv",
                        index=False,
                        encoding=encoding,
                    )
                if cleanup_errors:
                    raise RuntimeError(
                        "Locked-test model cleanup failed: "
                        + "; ".join(
                            f"{type(item).__name__}: {item}" for item in cleanup_errors
                        )
                    ) from cleanup_errors[0]
            model_watchdog.assert_safe()

        test_global_watchdog.stop()
        test_global_watchdog.assert_safe()
        test_global_gpu_guard = _global_gpu_guard_summary(
            test_global_watchdog.history, compute_audit
        )
        pd.DataFrame.from_records(test_global_watchdog.history).to_csv(
            test_dir / "global_gpu_memory_monitor.csv",
            index=False,
            encoding=encoding,
        )
        write_json(
            test_global_gpu_guard, test_dir / "global_gpu_memory_guard_summary.json"
        )
        test_gpu_guard = _gpu_guard_summary(
            test_gpu_memory_history,
            compute_audit,
            expected_pairs=[
                (str(selection["algorithm"]), int(selection["training_seed"]))
                for selection in selections
            ],
        )
        write_json(test_gpu_guard, test_dir / "gpu_memory_guard_summary.json")
        baseline_env = legacy._make_env(
            scenario, fast_bundle, rl_config, scenario.test_starts
        )
        phase_started = time.perf_counter()
        try:
            trajectories.append(
                rollout_historical_baseline(
                    scenario,
                    fast_bundle,
                    start_positions=scenario.test_starts,
                    preferences=preferences,
                )
            )
            trajectories.append(
                rollout_simple_baseline(
                    "KeepPrevious",
                    baseline_env,
                    start_positions=scenario.test_starts,
                    preferences=preferences,
                )
            )
            for seed in tuple(
                int(value) for value in rl_config["random_baseline_seeds"]
            ):
                trajectories.append(
                    rollout_simple_baseline(
                        "RandomFeasible",
                        baseline_env,
                        start_positions=scenario.test_starts,
                        preferences=preferences,
                        seed=seed,
                    )
                )
            runtime.append(
                {
                    "phase": "locked_test_transparent_baselines",
                    "method": "|".join(BASELINE_METHODS),
                    "seed": "registered",
                    "wall_seconds": time.perf_counter() - phase_started,
                }
            )
        finally:
            baseline_env.close()

        trajectory = pd.concat(trajectories, ignore_index=True)
        episode_summary = summarize_episodes(trajectory)
        fixed_panel = legacy._audit_fixed_test_panel(
            episode_summary,
            expected_starts=scenario.test_starts,
            preferences=preferences,
        )
        action_audit = legacy._audit_action_domain(trajectory, scenario)
        points, metrics, union_front = algorithm_multiobjective_summary(
            episode_summary,
            objective_low=scenario.objective_low,
            objective_high=scenario.objective_high,
            reference_point=tuple(rl_config["hypervolume_reference"]),
        )
        seed_metrics = _seedwise_metrics(
            points, union_front, tuple(rl_config["hypervolume_reference"])
        )
        failures = _failure_cases(episode_summary)
        support_summary = (
            episode_summary.groupby(["method", "training_seed", "preference_TN"], observed=True)
            .agg(
                support_valid_rate_mean=("support_valid_rate", "mean"),
                support_valid_rate_sd=("support_valid_rate", "std"),
                repair_rate_mean=("repair_rate", "mean"),
                repair_rate_sd=("repair_rate", "std"),
                episode_count=("episode_id", "size"),
            )
            .reset_index()
        )
        tables = {
            "test_trajectories.csv": trajectory,
            "test_episode_summary.csv": episode_summary,
            "test_pareto_points.csv": points,
            "test_algorithm_metrics.csv": metrics,
            "test_seedwise_metrics.csv": seed_metrics,
            "test_empirical_union_front_normalized.csv": union_front,
            "test_failure_cases_vs_random_feasible.csv": failures,
            "test_support_repair_summary.csv": support_summary,
            "test_runtime.csv": pd.DataFrame.from_records(runtime),
        }
        for name, table in tables.items():
            table.to_csv(test_dir / name, index=False, encoding=encoding)
        write_json(fixed_panel, test_dir / "fixed_test_panel_audit.json")
        write_json(action_audit, test_dir / "action_domain_audit.json")
        output_hashes = {
            path.name: sha256_file(path)
            for path in sorted(test_dir.iterdir())
            if path.is_file() and path.name not in {"TEST_ACCESS.json", "test_manifest.json"}
        }
        test_manifest = {
            "stage": "formal_rl_locked_test_once",
            "status": "completed",
            "completion_scope": "test_outputs",
            "run_state_authority": "../run_status.json",
            "started_utc": access_started,
            "completed_utc": _utc_now(),
            "elapsed_seconds": time.perf_counter() - started,
            "development_fingerprint": manifest["fingerprint"],
            "test_access_attempts": 1,
            "repeat_access_allowed": False,
            "all_validation_selections_frozen_before_access": True,
            "algorithms": list(algorithms),
            "training_seeds": list(rl_config["training_seeds"]),
            "random_baseline_seeds": list(rl_config["random_baseline_seeds"]),
            "baselines": list(BASELINE_METHODS),
            "preference_weights_TN": list(preferences),
            "hypervolume_reference_normalized": list(rl_config["hypervolume_reference"]),
            "IGD_plus_reference": "one-shot empirical non-dominated union of all frozen methods",
            "proxy_simulation_only": True,
            "upstream_proxy_independent_test": False,
            "plant_control_claim_authorized": False,
            "compute_device": str(rl_config["device"]),
            "gpu_memory_guard": test_gpu_guard,
            "global_gpu_memory_guard": test_global_gpu_guard,
            "outputs": output_hashes,
        }
        write_json(test_manifest, test_dir / "test_manifest.json")
        write_json(
            {
                "status": "TEST_ACCESS_COMPLETED_AND_RESEALED",
                "started_utc": access_started,
                "completed_utc": _utc_now(),
                "development_fingerprint": manifest["fingerprint"],
                "repeat_access_allowed": False,
                "test_manifest": "test_manifest.json",
                "test_accessed": True,
                "test_access_attempts": 1,
            },
            access_path,
        )
        write_json(
            {
                "status": "FORMAL_RL_COMPLETE_TEST_USED_ONCE",
                "development_fingerprint": manifest["fingerprint"],
                "test_accessed": True,
                "test_access_attempts": 1,
                "test_completed_utc": _utc_now(),
            },
            status_path,
        )
        print(
            json.dumps(
                {
                    "status": "formal_rl_complete_test_used_once",
                    "run_dir": str(run_dir),
                    "test_dir": str(test_dir),
                    "elapsed_seconds": test_manifest["elapsed_seconds"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        if "test_global_watchdog" in locals():
            test_global_watchdog.request_stop()
        if test_access_committed:
            failed_utc = _utc_now()
            failure_error = f"{type(exc).__name__}: {exc}"
            access_state_write_error: str | None = None
            try:
                write_json(
                    {
                        "status": "TEST_ACCESS_FAILED_NO_RETRY_ALLOWED",
                        "started_utc": access_started,
                        "failed_utc": failed_utc,
                        "test_accessed": True,
                        "test_access_attempts": 1,
                        "repeat_access_allowed": False,
                        "error": failure_error,
                    },
                    access_path,
                )
            except BaseException as state_exc:  # pragma: no cover - disk failure path
                access_state_write_error = f"{type(state_exc).__name__}: {state_exc}"
            failure_status = {
                "status": "FORMAL_RL_TEST_FAILED_NO_RETRY_ALLOWED",
                "development_fingerprint": manifest["fingerprint"],
                "test_accessed": True,
                "test_access_attempts": 1,
                "test_started_utc": access_started,
                "test_failed_utc": failed_utc,
                "error": failure_error,
                "access_state_write_error": access_state_write_error,
            }
            run_state_write_error: str | None = None
            try:
                write_json(failure_status, status_path)
            except BaseException as state_exc:  # pragma: no cover - disk failure path
                run_state_write_error = f"{type(state_exc).__name__}: {state_exc}"
            if "test_global_watchdog" in locals():
                test_global_watchdog.stop()
            if "preloaded_models" in locals():
                preloaded_models.clear()
            if compute_audit.get("device_type") == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            try:
                if (
                    "test_global_watchdog" in locals()
                    and test_global_watchdog.history
                ):
                    pd.DataFrame.from_records(test_global_watchdog.history).to_csv(
                        test_dir / "global_gpu_memory_monitor.csv",
                        index=False,
                        encoding=str(config["output"]["encoding"]),
                    )
                    global_unsafe = [
                        row
                        for row in test_global_watchdog.history
                        if float(row["global_used_fraction"])
                        >= float(
                            compute_audit.get("global_memory_abort_fraction", 1.0)
                        )
                    ]
                    if global_unsafe:
                        write_json(
                            {
                                "status": "GPU_MEMORY_GUARD_TRIGGERED",
                                "threshold": compute_audit.get(
                                    "global_memory_abort_fraction"
                                ),
                                "first_unsafe_sample": global_unsafe[0],
                            },
                            test_dir / "global_gpu_memory_breach.json",
                        )
                if "test_gpu_memory_history" in locals() and test_gpu_memory_history:
                    pd.DataFrame.from_records(test_gpu_memory_history).to_csv(
                        test_dir / "gpu_memory_monitor.csv",
                        index=False,
                        encoding=str(config["output"]["encoding"]),
                    )
                    unsafe_rows = [
                        row
                        for row in test_gpu_memory_history
                        if float(row["global_used_fraction"])
                        >= float(
                            compute_audit.get("global_memory_abort_fraction", 1.0)
                        )
                    ]
                    if unsafe_rows:
                        write_json(
                            {
                                "status": "GPU_MEMORY_GUARD_TRIGGERED",
                                "threshold": compute_audit.get(
                                    "global_memory_abort_fraction"
                                ),
                                "first_unsafe_sample": unsafe_rows[0],
                            },
                            test_dir / "gpu_memory_breach.json",
                        )
            except BaseException as diagnostic_exc:  # pragma: no cover - disk failure path
                failure_status["diagnostic_write_error"] = (
                    f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                )
                if run_state_write_error is None:
                    write_json(failure_status, status_path)
        else:
            if "test_global_watchdog" in locals():
                test_global_watchdog.stop()
            if "preloaded_models" in locals():
                preloaded_models.clear()
            if compute_audit.get("device_type") == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            if test_dir.exists() and not any(test_dir.iterdir()):
                test_dir.rmdir()
        raise


def main() -> int:
    args = _parse_args()
    if args.phase == "develop":
        if args.run_dir:
            raise ValueError("--run-dir is valid only for --phase test.")
        return _development(args)
    if args.output_dir or args.timesteps is not None or args.smoke:
        raise ValueError("--output-dir/--timesteps/--smoke are valid only for --phase develop.")
    return _test_once(args)


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    raise SystemExit(main())
