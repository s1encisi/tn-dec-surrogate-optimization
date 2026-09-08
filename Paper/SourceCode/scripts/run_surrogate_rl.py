from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import gc
import hashlib
from importlib import metadata
import json
import multiprocessing
from pathlib import Path
import platform
import sys
import time
from typing import Any, Sequence
import warnings

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


LGBM_FEATURE_WARNING = (
    r"X does not have valid feature names, but LGBMRegressor was fitted with feature names"
)
warnings.filterwarnings(
    "ignore",
    message=LGBM_FEATURE_WARNING,
    category=UserWarning,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from stable_baselines3 import PPO, SAC, TD3  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402
from stable_baselines3.common.monitor import Monitor  # noqa: E402
from stable_baselines3.common.noise import NormalActionNoise  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv  # noqa: E402
from pymoo.indicators.hv import HV  # noqa: E402

from taici.config import load_toml  # noqa: E402
from taici.fast_proxy import (  # noqa: E402
    FastExactProxyBundle,
    audit_fast_proxy_fidelity,
)
from taici.final_proxy import FinalProxyBundle  # noqa: E402
from taici.initial_dataset import load_initial_dataset  # noqa: E402
from taici.io import sha256_file, write_json  # noqa: E402
from taici.paired_random_windows import FEATURE_KEY  # noqa: E402
from taici.surrogate_rl import (  # noqa: E402
    ScenarioData,
    SurrogateRLEnv,
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
BASELINES = ("HistoricalObserved", "KeepPrevious", "RandomFeasible")
FORMAL_RL_CONFIG = "configs/formal_rl_rerun.toml"
LEGACY_RL_EVIDENCE_ROLE = "legacy_non_manuscript_reproduction_only"
BLUE, VERMILLION, GREEN = "#0072B2", "#D55E00", "#009E73"
PURPLE, BLACK, GRAY = "#CC79A7", "#000000", "#666666"
METHOD_STYLE = {
    "PPO": (BLUE, "o", "-"),
    "SAC": (VERMILLION, "s", "--"),
    "TD3": (GREEN, "^", "-."),
    "HistoricalObserved": (BLACK, "D", ":"),
    "KeepPrevious": (PURPLE, "P", (0, (3, 1, 1, 1))),
    "RandomFeasible": (GRAY, "X", (0, (1, 1))),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the legacy non-manuscript exact-proxy PPO/SAC/TD3 study; "
            "formal manuscript RL uses run_formal_rl_rerun.py."
        )
    )
    parser.add_argument("--config", default="configs/final_workflow.toml")
    parser.add_argument("--proxy-run-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run only PPO/seed 11 on a bounded legacy evaluation panel.",
    )
    parser.add_argument("--smoke-timesteps", type=int, default=None)
    parser.add_argument("--smoke-evaluation-episodes", type=int, default=2)
    return parser.parse_args()


def _validated_legacy_rl_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject an ambiguous legacy block that could be mistaken for manuscript evidence."""

    inputs = config.get("inputs")
    if not isinstance(inputs, Mapping):
        raise RuntimeError("Final workflow inputs must be a TOML table.")
    if inputs.get("formal_rl_config") != FORMAL_RL_CONFIG:
        raise RuntimeError("Final workflow must identify formal_rl_rerun.toml as manuscript RL.")

    if "rl" in config:
        raise RuntimeError(
            "Ambiguous [rl] is forbidden in final_workflow.toml; use the explicitly "
            "non-manuscript [legacy_rl_non_manuscript] block."
        )
    rl_config = config.get("legacy_rl_non_manuscript")
    if not isinstance(rl_config, Mapping):
        raise RuntimeError("Final workflow legacy_rl_non_manuscript block is missing.")
    if rl_config.get("evidence_role") != LEGACY_RL_EVIDENCE_ROLE:
        raise RuntimeError("The legacy RL block lacks its non-manuscript evidence role.")
    if bool(rl_config.get("manuscript_evidence_authorized")):
        raise RuntimeError("Legacy RL must never be authorized as manuscript evidence.")
    if rl_config.get("superseded_by") != FORMAL_RL_CONFIG:
        raise RuntimeError("Legacy RL must point to the formal manuscript RL configuration.")
    return dict(rl_config)


def _resolve_proxy_run(config: dict[str, Any], argument: str | None) -> Path:
    if argument:
        candidate = Path(argument)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    latest_path = PROJECT_ROOT / str(config["output"]["root"]) / "LATEST_PROXY.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "completed":
        raise RuntimeError("The latest final proxy run is not completed.")
    return PROJECT_ROOT / str(latest["run_dir"])


def _resolve_output_dir(
    proxy_run: Path, argument: str | None, *, smoke: bool, timestamp: str
) -> Path:
    if argument:
        candidate = Path(argument)
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return proxy_run / (f"rl_smoke_{timestamp}" if smoke else "rl")


def _environment() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "joblib",
        "pymoo",
        "matplotlib",
    )
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = "NOT_INSTALLED"
    return {
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "torch_threads_parent": str(torch.get_num_threads()),
        **versions,
    }


def _make_env(
    scenario: ScenarioData,
    bundle: Any,
    rl_config: dict[str, Any],
    starts: Sequence[int],
) -> SurrogateRLEnv:
    return SurrogateRLEnv(
        scenario,
        bundle,
        start_positions=tuple(int(value) for value in starts),
        support_penalty=float(rl_config["support_penalty"]),
        repair_penalty=float(rl_config["repair_penalty"]),
        invalid_penalty=float(rl_config["invalid_penalty"]),
    )


class _EnvFactory:
    """Picklable Windows-spawn factory for one exact-proxy training worker."""

    def __init__(
        self,
        scenario: ScenarioData,
        bundle: Any,
        rl_config: dict[str, Any],
        starts: Sequence[int],
        seed: int,
    ) -> None:
        self.scenario = scenario
        self.bundle = bundle
        self.rl_config = rl_config
        self.starts = tuple(int(value) for value in starts)
        self.seed = int(seed)

    def __call__(self) -> Monitor:
        torch.set_num_threads(1)
        env = _make_env(self.scenario, self.bundle, self.rl_config, self.starts)
        env.reset(seed=self.seed)
        return Monitor(env)


def _make_training_env(
    scenario: ScenarioData,
    bundle: Any,
    rl_config: dict[str, Any],
    *,
    seed: int,
    n_envs: int,
) -> Any:
    if n_envs == 1:
        env = _make_env(scenario, bundle, rl_config, scenario.train_starts)
        env.reset(seed=seed)
        return env
    factories = [
        _EnvFactory(
            scenario,
            bundle,
            rl_config,
            scenario.train_starts,
            seed + rank * 10_003,
        )
        for rank in range(n_envs)
    ]
    return SubprocVecEnv(factories, start_method="spawn")


def _limit_estimator_threads(bundle: FinalProxyBundle) -> dict[str, Any]:
    """Set inference-only parallelism to one without changing fitted parameters."""

    estimators = {"DEC": bundle.dec_model, **dict(bundle.tn_model.base_models)}
    changed: dict[str, list[str]] = {}
    skipped: dict[str, list[str]] = {}
    for name, estimator in estimators.items():
        if not hasattr(estimator, "get_params") or not hasattr(estimator, "set_params"):
            continue
        parameters = estimator.get_params(deep=True)
        keys = [key for key in parameters if key == "n_jobs" or key.endswith("__n_jobs")]
        for key in keys:
            try:
                estimator.set_params(**{key: 1})
                changed.setdefault(name, []).append(key)
            except Exception as exc:
                skipped.setdefault(name, []).append(f"{key}:{type(exc).__name__}")
    return {"changed": changed, "skipped_fitted_immutable": skipped}


def _thread_fidelity_audit(
    bundle: FinalProxyBundle,
    feature_set: Any,
) -> dict[str, Any]:
    tn_frame = feature_set.bundles["TN_out"].matrix(FEATURE_KEY)
    dec_frame = feature_set.bundles["DEC"].matrix(FEATURE_KEY)
    before_tn = bundle.predict_tn(tn_frame)
    before_dec = bundle.predict_dec(dec_frame)
    changed = _limit_estimator_threads(bundle)
    after_tn = bundle.predict_tn(tn_frame)
    after_dec = bundle.predict_dec(dec_frame)
    errors = {
        "TN_out": float(np.max(np.abs(before_tn - after_tn))),
        "DEC": float(np.max(np.abs(before_dec - after_dec))),
    }
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"Inference thread limiting changed proxy predictions: {errors}")
    return {
        "status": "passed",
        "operation": "set mutable fitted-estimator inference n_jobs to one",
        "operation_is_model_approximation": False,
        "components_removed": 0,
        "weights_changed": False,
        "estimators_with_thread_parameters_changed": changed,
        "audit_rows": len(tn_frame),
        "max_abs_prediction_error": errors,
        "final_test_uses_all_original_components_and_weights": True,
    }


def _action_perturbation_frames(
    scenario: ScenarioData,
    *,
    sampled_rows: int = 64,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build deterministic in-bound PPA/DO probes for off-observation fidelity."""

    available = np.asarray(scenario.train_starts, dtype=int)
    count = min(int(sampled_rows), len(available))
    selected = available[np.linspace(0, len(available) - 1, count, dtype=int)]
    patterns = np.asarray(
        [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0), (0.0, 0.0)],
        dtype=float,
    )
    records: list[pd.Series] = []
    for position in selected:
        row = scenario.frame.iloc[int(position)]
        previous = np.array([row["PPA_recent"], row["DO_recent"]], dtype=float)
        for pattern in patterns:
            action = np.clip(
                previous + pattern * scenario.action_rate,
                scenario.action_low,
                scenario.action_high,
            )
            features = row.loc[list(scenario.union_feature_names)].copy()
            features["PPA"] = float(
                np.mean([row["PPA_older"], row["PPA_recent"], action[0]])
            )
            features["DO"] = float(
                np.mean([row["DO_older"], row["DO_recent"], action[1]])
            )
            records.append(features)
    tn_frame = pd.DataFrame.from_records(
        records,
        columns=list(scenario.union_feature_names),
    )
    dec_frame = tn_frame.loc[:, list(scenario.union_feature_names)].copy()
    return tn_frame, dec_frame, {
        "source_training_rows": count,
        "action_patterns_per_row": len(patterns),
        "probe_rows": len(tn_frame),
        "construction": "previous_action_plus_signed_q90_rate_then_q05_q95_clip",
        "uses_target_or_output_history": False,
    }


def _ppo_batch_settings(total_timesteps: int, n_envs: int) -> tuple[int, int]:
    if total_timesteps == 8_000 and n_envs == 4:
        return 250, 100
    per_env = max(2, total_timesteps // n_envs)
    candidates = [
        value
        for value in range(min(128, per_env), 1, -1)
        if total_timesteps % (value * n_envs) == 0
    ]
    n_steps = candidates[0] if candidates else min(128, per_env)
    batches = [value for value in range(min(128, n_steps * n_envs), 1, -1) if (n_steps * n_envs) % value == 0]
    return n_steps, batches[0] if batches else n_steps * n_envs


def _build_model(
    algorithm: str,
    env: Any,
    *,
    seed: int,
    total_timesteps: int,
    n_envs: int,
    hidden_layers: Sequence[int],
    device: str,
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
        n_steps, batch_size = _ppo_batch_settings(total_timesteps, n_envs)
        return PPO(
            **common,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=5,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
        )
    learning_starts = min(500, max(n_envs, total_timesteps // 4))
    off_policy = {
        **common,
        "buffer_size": max(2_000, min(20_000, total_timesteps * 2)),
        "learning_starts": learning_starts,
        "batch_size": min(128, max(16, total_timesteps // 4)),
        "gamma": 0.99,
        "tau": 0.005,
        "train_freq": 1,
        "gradient_steps": -1,
    }
    if algorithm == "SAC":
        return SAC(**off_policy, learning_rate=3e-4, ent_coef="auto")
    if algorithm == "TD3":
        noise = NormalActionNoise(mean=np.zeros(2), sigma=np.full(2, 0.1))
        return TD3(
            **off_policy,
            learning_rate=1e-3,
            action_noise=noise,
            policy_delay=2,
        )
    raise ValueError(f"Unsupported algorithm: {algorithm}")


class ValidationCheckpointCallback(BaseCallback):
    """Select checkpoints only from fixed 2025-H1 starts and preferences."""

    def __init__(
        self,
        *,
        algorithm: str,
        training_seed: int,
        validation_env: SurrogateRLEnv,
        validation_starts: Sequence[int],
        preferences: Sequence[float],
        evaluation_frequency: int,
        model_dir: Path,
        objective_low: np.ndarray,
        objective_high: np.ndarray,
        reference_point: Sequence[float],
    ) -> None:
        super().__init__(verbose=0)
        self.algorithm = algorithm
        self.training_seed = int(training_seed)
        self.validation_env = validation_env
        self.validation_starts = tuple(int(value) for value in validation_starts)
        self.preferences = tuple(float(value) for value in preferences)
        self.evaluation_frequency = int(evaluation_frequency)
        self.model_dir = Path(model_dir)
        self.objective_low = np.asarray(objective_low, dtype=float)
        self.objective_high = np.asarray(objective_high, dtype=float)
        self.reference_point = np.asarray(reference_point, dtype=float)
        if self.objective_low.shape != (2,) or self.objective_high.shape != (2,):
            raise ValueError("Validation objective scaling must contain TN and DEC bounds.")
        if self.reference_point.shape != (2,):
            raise ValueError("The validation hypervolume reference point must have two values.")
        self.next_evaluation = self.evaluation_frequency
        self.pending_evaluation = False
        self.evaluated_steps: set[int] = set()
        self.history: list[dict[str, Any]] = []
        self.best_hv = -np.inf
        self.best_support_valid_rate = -np.inf
        self.best_step = -1
        self.best_model_path = self.model_dir / "best_model.zip"

    def _evaluate(self) -> None:
        step = int(self.num_timesteps)
        if step in self.evaluated_steps:
            return
        started = time.perf_counter()
        trajectory = rollout_policy(
            self.model,
            self.validation_env,
            algorithm=self.algorithm,
            training_seed=self.training_seed,
            start_positions=self.validation_starts,
            preferences=self.preferences,
        )
        episodes = summarize_episodes(trajectory)
        validation_points = (
            episodes.groupby("preference_TN", observed=True)
            .agg(
                mean_TN_out=("mean_TN_out", "mean"),
                mean_DEC=("mean_DEC", "mean"),
            )
            .reset_index()
            .sort_values("preference_TN")
        )
        if len(validation_points) != len(self.preferences):
            raise RuntimeError("A validation checkpoint lacks one or more preference points.")
        raw_objectives = validation_points[["mean_TN_out", "mean_DEC"]].to_numpy(float)
        normalized = (raw_objectives - self.objective_low) / (
            self.objective_high - self.objective_low
        )
        front = normalized[non_dominated_mask(normalized)]
        inside_reference = np.all(front < self.reference_point, axis=1)
        validation_hv = (
            float(HV(ref_point=self.reference_point)(front[inside_reference]))
            if inside_reference.any()
            else 0.0
        )
        support_valid_rate = float(episodes["support_valid_rate"].mean())
        self.model.save(self.model_dir / f"checkpoint_{step:08d}_steps.zip")
        hv_improved = validation_hv > self.best_hv + 1e-12
        hv_tied = abs(validation_hv - self.best_hv) <= 1e-12
        support_improved = support_valid_rate > self.best_support_valid_rate + 1e-12
        improved = hv_improved or (hv_tied and support_improved)
        if improved:
            self.best_hv = validation_hv
            self.best_support_valid_rate = support_valid_rate
            self.best_step = step
            self.model.save(self.best_model_path)
        elapsed = time.perf_counter() - started
        for preference, group in episodes.groupby("preference_TN", observed=True):
            self.history.append(
                {
                    "algorithm": self.algorithm,
                    "training_seed": self.training_seed,
                    "checkpoint_step": step,
                    "preference_TN": float(preference),
                    "mean_cumulative_reward": float(group["cumulative_reward"].mean()),
                    "mean_TN_out": float(group["mean_TN_out"].mean()),
                    "mean_DEC": float(group["mean_DEC"].mean()),
                    "mean_support_valid_rate": float(group["support_valid_rate"].mean()),
                    "mean_repair_rate": float(group["repair_rate"].mean()),
                    "n_episodes": int(len(group)),
                    "validation_HV": validation_hv,
                    "validation_non_dominated_points": int(len(front)),
                    "validation_mean_support_valid_rate": support_valid_rate,
                    "selection_score": validation_hv,
                    "selected_as_best_at_this_step": bool(improved),
                    "evaluation_seconds": elapsed,
                    "split": "validation_2025_H1",
                    "test_accessed": False,
                    "selection_rule": (
                        "validation_HV_desc_then_support_valid_rate_desc_then_earlier_step"
                    ),
                    "cumulative_reward_role": "descriptive_not_checkpoint_selection_score",
                }
            )
        self.evaluated_steps.add(step)

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_evaluation:
            self.pending_evaluation = True
            while self.next_evaluation <= self.num_timesteps:
                self.next_evaluation += self.evaluation_frequency
        return True

    def _on_rollout_start(self) -> None:
        if self.pending_evaluation:
            self._evaluate()
            self.pending_evaluation = False

    def _on_training_end(self) -> None:
        if int(self.num_timesteps) not in self.evaluated_steps:
            self._evaluate()
        self.pending_evaluation = False


def _scenario_contract(scenario: ScenarioData, rl_config: dict[str, Any]) -> dict[str, Any]:
    dates = pd.DatetimeIndex(pd.to_datetime(scenario.frame["Date"])).normalize()

    def episode_dates(starts: Sequence[int]) -> list[str]:
        return [dates[int(value)].date().isoformat() for value in starts]

    return {
        "analysis_role": "historical_support_domain_constrained_surrogate_simulation",
        "proxy_simulation_only": True,
        "plant_control_claim_authorized": False,
        "action_variables": ["PPA", "DO"],
        "action_role": "virtual_fast_controls_not_confirmed_plant_setpoints",
        "slow_state": "MLSS",
        "slow_state_role": "historical_exogenous_replay_not_action",
        "feature_version": FEATURE_KEY,
        "union_feature_names": list(scenario.union_feature_names),
        "observation_feature_names": [*scenario.observation_feature_names, "preference_TN"],
        "preference_weights_TN": list(scenario.preferences),
        "preference_weights_DEC": [1.0 - value for value in scenario.preferences],
        "episode_horizon_days": scenario.episode_horizon,
        "scenario_rows": len(scenario.frame),
        "action_low_training_q05": scenario.action_low.tolist(),
        "action_high_training_q95": scenario.action_high.tolist(),
        "action_rate_training_q90": scenario.action_rate.tolist(),
        "objective_low_training_q05": scenario.objective_low.tolist(),
        "objective_high_training_q95": scenario.objective_high.tolist(),
        "support_threshold": scenario.support_threshold,
        "support_neighbors": int(rl_config["support_neighbors"]),
        "splits": {
            "train_2023_2024": {
                "configured_start": str(rl_config["train_start"]),
                "configured_end": str(rl_config["train_end"]),
                "episode_starts": episode_dates(scenario.train_starts),
            },
            "validation_2025_H1": {
                "configured_start": str(rl_config["validation_start"]),
                "configured_end": str(rl_config["validation_end"]),
                "episode_starts": episode_dates(scenario.validation_starts),
                "role": "checkpoint_selection_only",
            },
            "test_2025_H2": {
                "configured_start": str(rl_config["test_start"]),
                "configured_end": str(rl_config["test_end"]),
                "episode_starts": episode_dates(scenario.test_starts),
                "role": "final_evaluation_after_all_checkpoints_frozen",
            },
        },
        "metadata": dict(scenario.metadata),
    }


def _environment_audit(
    scenario: ScenarioData,
    bundle: Any,
    rl_config: dict[str, Any],
) -> dict[str, Any]:
    env = _make_env(scenario, bundle, rl_config, scenario.train_starts)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        warnings.filterwarnings(
            "ignore",
            message=LGBM_FEATURE_WARNING,
            category=UserWarning,
        )
        check_env(env, warn=True, skip_render_check=True)
    first = scenario.frame.iloc[scenario.train_starts[0]]
    reconstructed = {
        "PPA": float(
            np.mean(
                [
                    first["PPA_older"],
                    first["PPA_recent"],
                    first["PPA_historical_action"],
                ]
            )
        ),
        "DO": float(
            np.mean(
                [
                    first["DO_older"],
                    first["DO_recent"],
                    first["DO_historical_action"],
                ]
            )
        ),
    }
    errors = {
        "PPA": abs(reconstructed["PPA"] - float(first["PPA"])),
        "DO": abs(reconstructed["DO"] - float(first["DO"])),
    }
    if max(errors.values()) > 1e-10:
        raise RuntimeError(f"Model-input queue reconstruction failed: {errors}")
    split_sets = [
        set(scenario.train_starts),
        set(scenario.validation_starts),
        set(scenario.test_starts),
    ]
    for left in range(3):
        for right in range(left):
            if split_sets[left].intersection(split_sets[right]):
                raise RuntimeError("Scenario episode-start splits overlap.")
    observation, info = env.reset(seed=20260823)
    if observation.shape != (15,) or not np.isfinite(observation).all():
        raise RuntimeError("Environment reset returned an invalid observation.")
    env.close()
    return {
        "stable_baselines3_check_env": "passed",
        "check_env_warnings": list(dict.fromkeys(str(item.message) for item in caught)),
        "observation_shape": list(observation.shape),
        "action_shape": [2],
        "reset_info_keys": sorted(info),
        "model_input_reconstruction_max_abs_error": max(errors.values()),
        "model_input_reconstruction_errors": errors,
        "split_episode_starts_are_disjoint": True,
        "MLSS_is_action": False,
        "PPA_DO_are_confirmed_plant_setpoints": False,
        "proxy_simulation_only": True,
    }


def _benchmark_vector_env(
    scenario: ScenarioData,
    bundle: Any,
    rl_config: dict[str, Any],
    *,
    n_envs: int,
    batches: int,
) -> dict[str, Any]:
    if n_envs <= 1:
        return {"status": "not_run_for_single_env_smoke", "n_envs": n_envs}
    started = time.perf_counter()
    env = _make_training_env(
        scenario,
        bundle,
        rl_config,
        seed=20260823,
        n_envs=n_envs,
    )
    startup_seconds = time.perf_counter() - started
    env.reset()
    actions = np.zeros((n_envs, 2), dtype=np.float32)
    durations: list[float] = []
    for _ in range(int(batches)):
        batch_started = time.perf_counter()
        env.step(actions)
        durations.append(time.perf_counter() - batch_started)
    env.close()
    values = np.asarray(durations, dtype=float)
    transitions = int(batches) * n_envs
    return {
        "status": "completed",
        "implementation": "SubprocVecEnv_start_method_spawn_exact_full_proxy",
        "n_envs": n_envs,
        "batches": int(batches),
        "transitions": transitions,
        "startup_seconds": startup_seconds,
        "mean_seconds_per_vector_batch": float(values.mean()),
        "p50_seconds_per_vector_batch": float(np.quantile(values, 0.5)),
        "p95_seconds_per_vector_batch": float(np.quantile(values, 0.95)),
        "transitions_per_second": float(transitions / values.sum()),
        "proxy_approximation_used": False,
    }


def _audit_fixed_test_panel(
    episode_summary: pd.DataFrame,
    *,
    expected_starts: Sequence[int],
    preferences: Sequence[float],
) -> dict[str, Any]:
    expected = {
        (float(preference), int(start))
        for preference in preferences
        for start in expected_starts
    }
    records: list[dict[str, Any]] = []
    for (method, seed), group in episode_summary.groupby(
        ["method", "training_seed"], observed=True
    ):
        observed = set(
            zip(
                group["preference_TN"].astype(float),
                group["episode_start_position"].astype(int),
                strict=True,
            )
        )
        records.append(
            {
                "method": str(method),
                "training_seed": int(seed),
                "expected_episodes": len(expected),
                "observed_episodes": len(observed),
                "matches_fixed_panel": observed == expected,
            }
        )
    if not records or not all(record["matches_fixed_panel"] for record in records):
        raise RuntimeError("At least one method did not use the fixed test episode panel.")
    payload = "\n".join(f"{preference:.2f},{start}" for preference, start in sorted(expected))
    return {
        "status": "passed",
        "test_access_role": "final_evaluation_after_all_validation_checkpoints_frozen",
        "fixed_panel_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "preference_start_pairs": len(expected),
        "groups": records,
    }


def _audit_action_domain(
    trajectories: pd.DataFrame,
    scenario: ScenarioData,
) -> dict[str, Any]:
    if not np.isfinite(
        trajectories[["applied_PPA", "applied_DO"]].to_numpy(float)
    ).all():
        raise RuntimeError("A final-test trajectory contains a non-finite applied action.")
    controlled = trajectories.loc[
        ~trajectories["method"].eq("HistoricalObserved")
    ].copy()
    tolerance = 1e-10
    ppa_valid = controlled["applied_PPA"].between(
        scenario.action_low[0] - tolerance,
        scenario.action_high[0] + tolerance,
    )
    do_valid = controlled["applied_DO"].between(
        scenario.action_low[1] - tolerance,
        scenario.action_high[1] + tolerance,
    )
    if not bool((ppa_valid & do_valid).all()):
        raise RuntimeError("An environment-controlled test action is outside q05-q95 bounds.")
    historical = trajectories.loc[trajectories["method"].eq("HistoricalObserved")]
    historical_valid = historical["applied_PPA"].between(
        scenario.action_low[0] - tolerance,
        scenario.action_high[0] + tolerance,
    ) & historical["applied_DO"].between(
        scenario.action_low[1] - tolerance,
        scenario.action_high[1] + tolerance,
    )
    anchor_repairs = controlled.get(
        "initial_anchor_repair_PPA",
        pd.Series(0.0, index=controlled.index),
    ).fillna(0.0) + controlled.get(
        "initial_anchor_repair_DO",
        pd.Series(0.0, index=controlled.index),
    ).fillna(0.0)
    return {
        "status": "passed",
        "controlled_methods": sorted(controlled["method"].unique().tolist()),
        "controlled_action_rows": len(controlled),
        "controlled_out_of_domain_rows": int((~(ppa_valid & do_valid)).sum()),
        "controlled_applied_PPA_min_max": [
            float(controlled["applied_PPA"].min()),
            float(controlled["applied_PPA"].max()),
        ],
        "controlled_applied_DO_min_max": [
            float(controlled["applied_DO"].min()),
            float(controlled["applied_DO"].max()),
        ],
        "registered_action_low": scenario.action_low.tolist(),
        "registered_action_high": scenario.action_high.tolist(),
        "rows_with_initial_anchor_repair": int((anchor_repairs > tolerance).sum()),
        "historical_observed_rows": len(historical),
        "historical_observed_outside_experimental_domain_rows": int(
            (~historical_valid).sum()
        ),
        "historical_observed_role": (
            "behavior_reference_not_an_environment_controlled_feasible_policy"
        ),
    }


def _pareto_figure(points: pd.DataFrame, output_base: Path) -> list[Path]:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    fig, ax = plt.subplots(figsize=(7.4, 5.6), layout="constrained")
    for method, (color, marker, line_style) in METHOD_STYLE.items():
        group = points.loc[points["method"].eq(method)].copy()
        if group.empty:
            continue
        for _, seed_group in group.groupby("training_seed", observed=True):
            seed_group = seed_group.sort_values("preference_TN")
            ax.plot(
                seed_group["mean_TN_out"],
                seed_group["mean_DEC"],
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.2,
                markersize=5,
                alpha=0.72,
            )
        ax.scatter([], [], color=color, marker=marker, label=method)
    front = (
        points.loc[points["non_dominated_global"]]
        .drop_duplicates(subset=["mean_TN_out", "mean_DEC"])
        .sort_values("mean_TN_out")
    )
    ax.plot(
        front["mean_TN_out"],
        front["mean_DEC"],
        color=BLACK,
        linewidth=2.0,
        linestyle="--",
        label="Empirical union Pareto front",
        zorder=1,
    )
    ax.set_xlabel("Mean proxy-predicted TN_out (mg/L; lower is better)")
    ax.set_ylabel("Mean proxy-predicted DEC (kWh/d; lower is better)")
    ax.set_title("A  Test-period proxy Pareto comparison", loc="left")
    ax.grid(color="#E5E5E5", linewidth=0.5)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=300, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def _indicator_figure(metrics: pd.DataFrame, output_base: Path) -> list[Path]:
    ordered = metrics.sort_values("hypervolume", ascending=False).reset_index(drop=True)
    colors = [METHOD_STYLE[str(method)][0] for method in ordered["method"]]
    hatches = ["", "//", "..", "xx", "++", "\\\\"][: len(ordered)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout="constrained")
    positions = np.arange(len(ordered))
    settings = (
        (axes[0], "hypervolume", "A  Hypervolume (higher is better)", "HV"),
        (
            axes[1],
            "IGD_plus_to_empirical_union_front",
            "B  IGD+ to empirical union front (lower is better)",
            "IGD+",
        ),
    )
    for ax, column, title, label in settings:
        bars = ax.bar(
            positions,
            ordered[column],
            color=colors,
            edgecolor=BLACK,
            linewidth=0.6,
        )
        for bar, hatch in zip(bars, hatches, strict=False):
            bar.set_hatch(hatch)
        ax.set_xticks(positions, ordered["method"], rotation=30, ha="right")
        ax.set_ylabel(label)
        ax.set_title(title, loc="left")
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#E5E5E5", linewidth=0.5)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=300, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def _action_figure(trajectories: pd.DataFrame, output_base: Path) -> list[Path]:
    methods = ("PPO", "SAC", "TD3", "HistoricalObserved")
    selected = trajectories.loc[trajectories["method"].isin(methods)]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), layout="constrained")
    for method in methods:
        color, marker, line_style = METHOD_STYLE[method]
        group = selected.loc[selected["method"].eq(method)]
        if group.empty:
            continue
        for ax, column, label in (
            (axes[0], "applied_PPA", "Applied virtual PPA"),
            (axes[1], "applied_DO", "Applied virtual DO"),
        ):
            summary = group.groupby("step", observed=True)[column].agg(["mean", "std"])
            ax.plot(
                summary.index,
                summary["mean"],
                color=color,
                marker=marker,
                linestyle=line_style,
                linewidth=1.5,
                markersize=4,
                label=method,
            )
            if summary["std"].notna().any():
                ax.fill_between(
                    summary.index,
                    summary["mean"] - summary["std"].fillna(0),
                    summary["mean"] + summary["std"].fillna(0),
                    color=color,
                    alpha=0.08,
                )
            ax.set_xlabel("Episode day")
            ax.set_ylabel(label)
            ax.grid(color="#E5E5E5", linewidth=0.5)
    axes[0].set_title("A  PPA trajectory (mean ± descriptive SD)", loc="left")
    axes[1].set_title("B  DO trajectory (mean ± descriptive SD)", loc="left")
    axes[1].legend(frameon=False, fontsize=8)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    fig.savefig(paths[0], dpi=300, facecolor="white")
    fig.savefig(paths[1], facecolor="white")
    plt.close(fig)
    return paths


def _figure_manifest() -> dict[str, Any]:
    return {
        "general_figure_standard": "provisional_publication_ready_no_target_journal_claim",
        "palette": "Okabe-Ito-derived colors plus marker, line-style and hatch redundancy",
        "figures": {
            "Fig_RL_Pareto": {
                "caption": (
                    "Mean seven-day TN_out and DEC proxy predictions for fixed 2025-H2 "
                    "episodes. Each RL point is one algorithm, seed and registered preference. "
                    "The dashed line is an empirical proxy front, not a true plant front."
                ),
                "alt_text": (
                    "Scatter plot comparing PPO, SAC, TD3 and three baselines on lower-is-better "
                    "TN_out and DEC proxy objectives; marker shapes also identify methods."
                ),
            },
            "Fig_RL_Indicators": {
                "caption": (
                    "HV and IGD+ after training-period objective normalization. IGD+ uses the "
                    "empirical union front, not a known true front."
                ),
                "alt_text": "Two bar charts compare methods separately by HV and IGD+.",
            },
            "Fig_RL_Actions": {
                "caption": (
                    "Virtual PPA and DO trajectories pooled across fixed test episodes and "
                    "preferences for PPO, SAC, TD3 and HistoricalObserved. KeepPrevious and "
                    "RandomFeasible remain reported in the metric and Pareto tables. Bands are "
                    "descriptive standard deviations, not confidence intervals; PPA is reported "
                    "in kg and is field-adjustable, whereas DO is a measured virtual scenario coordinate."
                ),
                "alt_text": "Two line charts show average PPA and DO over seven-day episodes.",
                "displayed_methods": ["PPO", "SAC", "TD3", "HistoricalObserved"],
                "methods_reported_elsewhere_not_displayed": [
                    "KeepPrevious",
                    "RandomFeasible",
                ],
                "action_unit_semantics": "PPA_kg_confirmed_DO_mg_per_L_measured",
            },
        },
        "underlying_data": ["test_trajectories.csv", "test_pareto_points.csv"],
        "plant_control_claim_authorized": False,
    }


def _collect_output_hashes(output_dir: Path) -> dict[str, str]:
    excluded = {output_dir / "manifest.json", output_dir / "run_status.json"}
    return {
        path.relative_to(output_dir).as_posix(): sha256_file(path)
        for path in sorted(output_dir.rglob("*"))
        if path.is_file() and path not in excluded
    }


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_toml(str(config_path))
    rl_config = _validated_legacy_rl_config(config)
    algorithms = tuple(str(value) for value in rl_config["algorithms"])
    seeds = tuple(int(value) for value in rl_config["training_seeds"])
    preferences = tuple(float(value) for value in rl_config["preference_weights"])
    if algorithms != ("PPO", "SAC", "TD3"):
        raise RuntimeError("The legacy algorithm matrix must be PPO, SAC and TD3.")
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise RuntimeError("The legacy protocol requires three distinct training seeds.")
    if preferences != (0.25, 0.50, 0.75):
        raise RuntimeError("Preference weights must be exactly 0.25, 0.50 and 0.75.")
    if args.smoke:
        algorithms, seeds = algorithms[:1], seeds[:1]
        total_timesteps = int(
            args.smoke_timesteps
            if args.smoke_timesteps is not None
            else rl_config["smoke_timesteps"]
        )
        if total_timesteps < 8:
            raise ValueError("Smoke training requires at least eight timesteps.")
    else:
        if args.smoke_timesteps is not None:
            raise ValueError("--smoke-timesteps is valid only with --smoke.")
        total_timesteps = int(rl_config["total_timesteps_per_algorithm_seed"])
        if total_timesteps != 8_000:
            raise RuntimeError("The frozen legacy reproduction budget is 8,000 steps per model.")
    training_n_envs = int(rl_config["training_parallel_envs"])
    if training_n_envs != 4:
        raise RuntimeError("The frozen exact-proxy training protocol requires four environments.")
    checkpoint_selection = str(rl_config["checkpoint_selection"])
    if checkpoint_selection != "validation_hypervolume":
        raise RuntimeError("Checkpoint selection must be validation_hypervolume.")

    proxy_run = _resolve_proxy_run(config, args.proxy_run_dir)
    proxy_manifest_path = proxy_run / "manifest.json"
    proxy_artifact_path = proxy_run / "final_proxy_bundle.joblib"
    proxy_manifest = json.loads(proxy_manifest_path.read_text(encoding="utf-8"))
    if proxy_manifest.get("status") != "completed":
        raise RuntimeError("The selected final proxy run is not completed.")
    original_bundle = joblib.load(proxy_artifact_path)
    if not isinstance(original_bundle, FinalProxyBundle):
        raise RuntimeError("The serialized final proxy has an unexpected type.")

    paired_config_path = PROJECT_ROOT / str(config["inputs"]["paired_config"])
    data_path = (PROJECT_ROOT / str(config["inputs"]["initial_dataset"])).resolve()
    initial = load_initial_dataset(data_path)
    phase8_status_path = PROJECT_ROOT / str(config["inputs"]["formal_plant_gate_status"])
    phase8_status = json.loads(phase8_status_path.read_text(encoding="utf-8"))
    if (
        phase8_status.get("status") != "BLOCKED"
        or bool(phase8_status.get("phase9_authorized"))
        or int(phase8_status.get("factory_optimizer_calls", -1)) != 0
    ):
        raise RuntimeError("The independent surrogate-RL study must preserve the blocked plant gate.")
    module_path = PROJECT_ROOT / "src" / "taici" / "surrogate_rl.py"
    fast_module_path = PROJECT_ROOT / "src" / "taici" / "fast_proxy.py"
    fingerprints = (
        sha256_file(config_path),
        sha256_file(paired_config_path),
        sha256_file(data_path),
        sha256_file(phase8_status_path),
        sha256_file(proxy_manifest_path),
        sha256_file(proxy_artifact_path),
        sha256_file(module_path),
        sha256_file(fast_module_path),
        sha256_file(Path(__file__)),
        f"mode={'legacy_smoke' if args.smoke else 'legacy_full_reproduction'}",
        f"timesteps={total_timesteps}",
        f"training_n_envs={training_n_envs}",
    )
    fingerprint = hashlib.sha256("\0".join(fingerprints).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = _resolve_output_dir(
        proxy_run,
        args.output_dir,
        smoke=bool(args.smoke),
        timestamp=timestamp,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "run_status.json"
    stage = (
        "surrogate_rl_legacy_smoke"
        if args.smoke
        else "surrogate_rl_legacy_non_manuscript_reproduction"
    )
    write_json(
        {
            "status": "RUNNING",
            "stage": stage,
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "evidence_role": LEGACY_RL_EVIDENCE_ROLE,
            "manuscript_evidence_authorized": False,
            "superseded_by": FORMAL_RL_CONFIG,
            "test_accessed": False,
        },
        status_path,
    )
    started = time.perf_counter()
    try:
        print("[surrogate-rl] loading the frozen paper initial dataset", flush=True)
        feature_set = initial.feature_set
        thread_audit = _thread_fidelity_audit(original_bundle, feature_set)
        write_json(thread_audit, output_dir / "inference_thread_fidelity_audit.json")

        print("[surrogate-rl] constructing and auditing the fast exact proxy", flush=True)
        fast_bundle = FastExactProxyBundle.from_frozen(original_bundle)
        full_panel_audit = audit_fast_proxy_fidelity(
            original_bundle,
            fast_bundle,
            feature_set.bundles["TN_out"].matrix(FEATURE_KEY),
            feature_set.bundles["DEC"].matrix(FEATURE_KEY),
            tolerance=1e-9,
        )
        scenario = build_scenario_data(feature_set, initial.frame, original_bundle, rl_config)
        probe_tn, probe_dec, probe_contract = _action_perturbation_frames(scenario)
        perturbation_audit = audit_fast_proxy_fidelity(
            original_bundle,
            fast_bundle,
            probe_tn,
            probe_dec,
            tolerance=1e-9,
        )
        test_access_started_utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        write_json(
            {
                "status": "PASSED",
                "full_1076_row_panel": full_panel_audit,
                "bounded_action_perturbations": perturbation_audit,
                "action_perturbation_contract": probe_contract,
                "training_uses_fast_exact_proxy": True,
                "validation_uses_fast_exact_proxy": True,
                "test_uses_fast_exact_proxy": True,
                "distillation_used": False,
                "surrogate_component_removed": False,
                "Huber_weights_changed": False,
            },
            output_dir / "fast_proxy_fidelity_audit.json",
        )
        validation_starts = scenario.validation_starts
        test_starts = scenario.test_starts
        if args.smoke:
            n_eval = int(args.smoke_evaluation_episodes)
            if n_eval < 1:
                raise ValueError("Smoke evaluation requires at least one episode start.")
            validation_starts = validation_starts[:n_eval]
            test_starts = test_starts[:n_eval]

        scenario_contract = _scenario_contract(scenario, rl_config)
        scenario_contract["execution_mode"] = (
            "legacy_smoke" if args.smoke else "legacy_full_reproduction"
        )
        scenario_contract["evidence_role"] = LEGACY_RL_EVIDENCE_ROLE
        scenario_contract["manuscript_evidence_authorized"] = False
        scenario_contract["superseded_by"] = FORMAL_RL_CONFIG
        scenario_contract["formal_plant_gate_status"] = "BLOCKED_preserved"
        scenario_contract["formal_plant_gate_source"] = str(
            config["inputs"]["formal_plant_gate_status"]
        )
        scenario_contract["formal_plant_gate_sha256"] = sha256_file(phase8_status_path)
        scenario_contract["phase9_authorized"] = False
        scenario_contract["study_is_separate_from_formal_plant_gate"] = True
        scenario_contract["training_vectorization"] = {
            "implementation": "SubprocVecEnv_spawn",
            "n_envs": training_n_envs,
            "full_exact_proxy_in_every_worker": True,
        }
        scenario_contract["evaluation_start_subset"] = {
            "validation_count": len(validation_starts),
            "test_count": len(test_starts),
            "legacy_full_uses_all_registered_starts": not args.smoke,
        }
        write_json(scenario_contract, output_dir / "scenario_contract.json")
        write_json(
            _environment_audit(scenario, fast_bundle, rl_config),
            output_dir / "environment_audit.json",
        )
        write_json(_environment(), output_dir / "environment.json")

        benchmark_env = _make_env(scenario, fast_bundle, rl_config, scenario.train_starts)
        single_benchmark = benchmark_proxy_step(
            benchmark_env,
            steps=5 if args.smoke else 30,
        )
        benchmark_env.close()
        vector_benchmark = _benchmark_vector_env(
            scenario,
            fast_bundle,
            rl_config,
            n_envs=training_n_envs,
            batches=3 if args.smoke else 8,
        )
        configured_steps = len(algorithms) * len(seeds) * total_timesteps
        speed_payload = {
            "single_environment_exact_fast_proxy": single_benchmark,
            "training_vector_environment": vector_benchmark,
            "configured_training_environment_steps": configured_steps,
            "selected_training_n_envs": training_n_envs,
            "selection_reason": (
                "four_spawned_exact_proxy_workers_reduce_single_row_inference_wall_time"
                if training_n_envs == 4
                else "bounded_smoke_avoids_process_startup"
            ),
            "policy_update_and_validation_test_overhead_excluded_from_simple_estimates": True,
            "proxy_approximation_used": False,
        }
        if vector_benchmark.get("status") == "completed":
            throughput = float(vector_benchmark["transitions_per_second"])
            speed_payload["estimated_training_seconds_from_measured_vector_throughput"] = (
                configured_steps / throughput
            )
        write_json(speed_payload, output_dir / "proxy_speed_benchmark.json")

        validation_history: list[dict[str, Any]] = []
        timing_records: list[dict[str, Any]] = []
        best_models: list[dict[str, Any]] = []
        history_path = output_dir / "validation_history.csv"
        encoding = str(config["output"]["encoding"])
        evaluation_frequency = min(int(rl_config["checkpoint_frequency"]), total_timesteps)
        print(
            "[surrogate-rl] training and selecting checkpoints; test remains sealed",
            flush=True,
        )
        for algorithm in algorithms:
            for seed in seeds:
                model_dir = models_dir / algorithm / f"seed_{seed}"
                model_dir.mkdir(parents=True, exist_ok=False)
                train_env = _make_training_env(
                    scenario,
                    fast_bundle,
                    rl_config,
                    seed=seed,
                    n_envs=training_n_envs,
                )
                validation_env = _make_env(
                    scenario,
                    fast_bundle,
                    rl_config,
                    validation_starts,
                )
                try:
                    model = _build_model(
                        algorithm,
                        train_env,
                        seed=seed,
                        total_timesteps=total_timesteps,
                        n_envs=training_n_envs,
                        hidden_layers=tuple(
                            int(value) for value in rl_config["policy_hidden_layers"]
                        ),
                        device=str(rl_config["device"]),
                    )
                    callback = ValidationCheckpointCallback(
                        algorithm=algorithm,
                        training_seed=seed,
                        validation_env=validation_env,
                        validation_starts=validation_starts,
                        preferences=preferences,
                        evaluation_frequency=evaluation_frequency,
                        model_dir=model_dir,
                        objective_low=scenario.objective_low,
                        objective_high=scenario.objective_high,
                        reference_point=tuple(
                            float(value) for value in rl_config["hypervolume_reference"]
                        ),
                    )
                    train_started = time.perf_counter()
                    model.learn(
                        total_timesteps=total_timesteps,
                        callback=callback,
                        progress_bar=False,
                    )
                    train_seconds = time.perf_counter() - train_started
                    validation_history.extend(callback.history)
                    pd.DataFrame.from_records(validation_history).to_csv(
                        history_path,
                        index=False,
                        encoding=encoding,
                    )
                    model.save(model_dir / "final_model.zip")
                    if callback.best_step < 0 or not callback.best_model_path.exists():
                        raise RuntimeError(
                            f"No validation checkpoint was selected for {algorithm}/{seed}."
                        )
                    timing_records.append(
                        {
                            "phase": "training_plus_validation",
                            "algorithm": algorithm,
                            "training_seed": seed,
                            "configured_timesteps": total_timesteps,
                            "actual_timesteps": int(model.num_timesteps),
                            "training_n_envs": training_n_envs,
                            "wall_seconds": train_seconds,
                            "best_validation_step": callback.best_step,
                            "best_validation_HV": callback.best_hv,
                            "best_validation_support_valid_rate": (
                                callback.best_support_valid_rate
                            ),
                            "test_accessed_during_selection": False,
                        }
                    )
                    best_models.append(
                        {
                            "algorithm": algorithm,
                            "training_seed": seed,
                            "model_path": callback.best_model_path,
                            "best_validation_step": callback.best_step,
                            "best_validation_HV": callback.best_hv,
                            "best_validation_support_valid_rate": (
                                callback.best_support_valid_rate
                            ),
                        }
                    )
                    del model
                    gc.collect()
                finally:
                    train_env.close()
                    validation_env.close()

        write_json(
            {
                "status": "RUNNING",
                "stage": stage,
                "created_utc": timestamp,
                "fingerprint": fingerprint,
                "all_validation_checkpoints_frozen": True,
                "test_accessed": True,
                "test_access_started_utc": test_access_started_utc,
            },
            status_path,
        )
        print(
            "[surrogate-rl] checkpoints frozen; starting common 2025-H2 test",
            flush=True,
        )
        trajectories: list[pd.DataFrame] = []
        for selection in best_models:
            algorithm = str(selection["algorithm"])
            seed = int(selection["training_seed"])
            model = ALGORITHM_CLASSES[algorithm].load(
                selection["model_path"],
                device=str(rl_config["device"]),
            )
            test_env = _make_env(scenario, fast_bundle, rl_config, test_starts)
            test_started = time.perf_counter()
            trajectories.append(
                rollout_policy(
                    model,
                    test_env,
                    algorithm=algorithm,
                    training_seed=seed,
                    start_positions=test_starts,
                    preferences=preferences,
                )
            )
            timing_records.append(
                {
                    "phase": "one_final_test_evaluation",
                    "algorithm": algorithm,
                    "training_seed": seed,
                    "configured_timesteps": total_timesteps,
                    "actual_timesteps": np.nan,
                    "training_n_envs": 1,
                    "wall_seconds": time.perf_counter() - test_started,
                    "best_validation_step": selection["best_validation_step"],
                    "best_validation_HV": selection["best_validation_HV"],
                    "best_validation_support_valid_rate": selection[
                        "best_validation_support_valid_rate"
                    ],
                    "test_accessed_during_selection": False,
                }
            )
            test_env.close()
            del model
            gc.collect()

        baseline_env = _make_env(scenario, fast_bundle, rl_config, test_starts)
        baseline_started = time.perf_counter()
        trajectories.append(
            rollout_historical_baseline(
                scenario,
                fast_bundle,
                start_positions=test_starts,
                preferences=preferences,
            )
        )
        trajectories.append(
            rollout_simple_baseline(
                "KeepPrevious",
                baseline_env,
                start_positions=test_starts,
                preferences=preferences,
            )
        )
        trajectories.append(
            rollout_simple_baseline(
                "RandomFeasible",
                baseline_env,
                start_positions=test_starts,
                preferences=preferences,
                seed=20260823,
            )
        )
        timing_records.append(
            {
                "phase": "three_baseline_final_test_evaluation",
                "algorithm": "|".join(BASELINES),
                "training_seed": 20260823,
                "configured_timesteps": 0,
                "actual_timesteps": 0,
                "training_n_envs": 1,
                "wall_seconds": time.perf_counter() - baseline_started,
                "best_validation_step": np.nan,
                "best_validation_HV": np.nan,
                "best_validation_support_valid_rate": np.nan,
                "test_accessed_during_selection": False,
            }
        )
        baseline_env.close()

        test_trajectory = pd.concat(trajectories, ignore_index=True)
        episode_summary = summarize_episodes(test_trajectory)
        fixed_panel_audit = _audit_fixed_test_panel(
            episode_summary,
            expected_starts=test_starts,
            preferences=preferences,
        )
        action_domain_audit = _audit_action_domain(test_trajectory, scenario)
        points, algorithm_metrics, union_front = algorithm_multiobjective_summary(
            episode_summary,
            objective_low=scenario.objective_low,
            objective_high=scenario.objective_high,
            reference_point=tuple(
                float(value) for value in rl_config["hypervolume_reference"]
            ),
        )
        write_json(fixed_panel_audit, output_dir / "fixed_test_panel_audit.json")
        write_json(action_domain_audit, output_dir / "action_domain_audit.json")
        tables = {
            "validation_history.csv": pd.DataFrame.from_records(validation_history),
            "runtime_summary.csv": pd.DataFrame.from_records(timing_records),
            "test_trajectories.csv": test_trajectory,
            "test_episode_summary.csv": episode_summary,
            "test_pareto_points.csv": points,
            "test_algorithm_metrics.csv": algorithm_metrics,
            "test_empirical_union_front_normalized.csv": union_front,
        }
        for filename, table in tables.items():
            table.to_csv(output_dir / filename, index=False, encoding=encoding)

        figure_paths: list[Path] = []
        figure_paths.extend(_pareto_figure(points, output_dir / "Fig_RL_Pareto"))
        figure_paths.extend(
            _indicator_figure(algorithm_metrics, output_dir / "Fig_RL_Indicators")
        )
        figure_paths.extend(
            _action_figure(test_trajectory, output_dir / "Fig_RL_Actions")
        )
        if len(figure_paths) != 6:
            raise RuntimeError("The RL figure export set is incomplete.")
        write_json(_figure_manifest(), output_dir / "figure_manifest.json")

        rl_metrics = algorithm_metrics.loc[
            algorithm_metrics["method_role"].eq("learned_RL_policy")
        ]
        best_rl = rl_metrics.loc[
            rl_metrics["headline_role"].eq("best_RL_by_HV"), "method"
        ]
        worst_rl = rl_metrics.loc[
            rl_metrics["headline_role"].eq(
                "worst_RL_posthoc_comparator_not_formal_baseline"
            ),
            "method",
        ]
        manifest = {
            "stage": stage,
            "status": "completed",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "elapsed_seconds": time.perf_counter() - started,
            "execution_mode": (
                "legacy_smoke" if args.smoke else "legacy_full_reproduction"
            ),
            "evidence_role": LEGACY_RL_EVIDENCE_ROLE,
            "manuscript_evidence_authorized": False,
            "superseded_by": FORMAL_RL_CONFIG,
            "analysis_role": config["scope"]["analysis_role"],
            "proxy_simulation_only": True,
            "plant_control_claim_authorized": False,
            "formal_plant_gate_status": "BLOCKED_preserved",
            "study_is_separate_from_formal_plant_gate": True,
            "independent_plant_optimization_evidence": False,
            "fast_proxy_role": (
                "numerically_equivalent_same_fitted_trees_all_15_components_not_distillation"
            ),
            "algorithms": list(algorithms),
            "training_seeds": list(seeds),
            "preference_weights_TN": list(preferences),
            "configured_timesteps_per_algorithm_seed": total_timesteps,
            "training_vector_environment": {
                "implementation": "SubprocVecEnv_spawn",
                "n_envs": training_n_envs,
            },
            "completed_policy_models": len(best_models),
            "legacy_policy_matrix_expected": 9,
            "checkpoint_selection_split": "2025-H1_validation_only",
            "checkpoint_selection_rule": checkpoint_selection,
            "final_evaluation_split": "2025-H2_after_all_checkpoints_frozen",
            "test_panel_shared_across_all_methods_and_seeds": True,
            "baselines": list(BASELINES),
            "algorithm_comparison": "Pareto_front_HV_and_IGDplus_no_composite_total_score",
            "hypervolume_reference_normalized": list(
                rl_config["hypervolume_reference"]
            ),
            "best_RL_by_test_HV": None if best_rl.empty else str(best_rl.iloc[0]),
            "worst_RL_posthoc_comparator": (
                None if worst_rl.empty else str(worst_rl.iloc[0])
            ),
            "worst_RL_is_formal_baseline": False,
            "inputs": {
                "config": sha256_file(config_path),
                "paired_config": sha256_file(paired_config_path),
                "initial_dataset": initial.file_sha256,
                "initial_dataset_manifest": sha256_file(initial.manifest_path),
                "initial_dataset_source": initial.source_sha256,
                "formal_plant_gate_status": sha256_file(phase8_status_path),
                "proxy_manifest": sha256_file(proxy_manifest_path),
                "proxy_artifact": sha256_file(proxy_artifact_path),
                "environment_module": sha256_file(module_path),
                "fast_proxy_module": sha256_file(fast_module_path),
                "runner": sha256_file(Path(__file__)),
            },
            "outputs": _collect_output_hashes(output_dir),
        }
        write_json(manifest, output_dir / "manifest.json")
        write_json(
            {
                "status": "COMPLETED",
                "stage": stage,
                "created_utc": timestamp,
                "completed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "manifest": "manifest.json",
                "test_accessed": True,
                "all_validation_checkpoints_frozen_before_test": True,
                "test_access_started_utc": test_access_started_utc,
            },
            status_path,
        )
        if not args.smoke:
            latest_path = (
                PROJECT_ROOT
                / str(config["output"]["root"])
                / "LATEST_LEGACY_RL_NON_MANUSCRIPT.json"
            )
            write_json(
                {
                    "stage": stage,
                    "status": "completed",
                    "evidence_role": LEGACY_RL_EVIDENCE_ROLE,
                    "manuscript_evidence_authorized": False,
                    "superseded_by": FORMAL_RL_CONFIG,
                    "formal_plant_gate_status": "BLOCKED_preserved",
                    "run_dir": proxy_run.relative_to(PROJECT_ROOT).as_posix(),
                    "rl_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                    "manifest": (output_dir / "manifest.json")
                    .relative_to(PROJECT_ROOT)
                    .as_posix(),
                    "fingerprint": fingerprint,
                },
                latest_path,
            )
        print(
            json.dumps(
                {
                    "status": "completed",
                    "mode": manifest["execution_mode"],
                    "rl_dir": str(output_dir),
                    "completed_policy_models": len(best_models),
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )
        return 0
    except BaseException as exc:
        write_json(
            {
                "status": "FAILED",
                "stage": stage,
                "created_utc": timestamp,
                "failed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_path,
        )
        raise


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())
