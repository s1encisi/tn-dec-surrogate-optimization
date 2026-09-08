from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT.parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.config import load_toml  # noqa: E402
from taici.initial_dataset import load_initial_dataset  # noqa: E402
from taici.io import sha256_file, write_json  # noqa: E402
from taici.p0_evidence import (  # noqa: E402
    FINAL_MODELS,
    build_joint_residual_library,
    parse_outer_folds,
    propagate_proxy_uncertainty,
    run_temporal_validation,
)
from taici.paired_random_windows import (  # noqa: E402
    BASE_MODELS,
    TARGETS,
    build_paired_model_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the post-hoc P0 temporal and proxy-uncertainty evidence audits."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "p0_evidence_audit.toml"
    module_path = PROJECT_ROOT / "src" / "taici" / "p0_evidence.py"
    config = load_toml(str(config_path))
    scope = config["scope"]
    temporal_config = config["temporal"]
    uncertainty_config = config["uncertainty"]
    if bool(scope["independent_test"]) or bool(scope["plant_control_claim_authorized"]):
        raise RuntimeError("P0 evidence audits cannot authorize independent-test or plant claims.")
    if tuple(temporal_config["targets"]) != TARGETS:
        raise RuntimeError("P0 temporal targets differ from the frozen target order.")
    if dict(temporal_config["final_models"]) != FINAL_MODELS:
        raise RuntimeError("P0 temporal models differ from the locked final models.")
    if tuple(config["ensemble"]["eligible_base_models"]) != BASE_MODELS:
        raise RuntimeError("P0 Huber components differ from the frozen base-model order.")

    data_path = (PROJECT_ROOT / str(config["input"]["initial_dataset"])).resolve()
    prediction_path = (
        PROJECT_ROOT / str(config["input"]["prediction_outer_predictions"])
    ).resolve()
    failure_path = (
        PROJECT_ROOT / str(config["input"]["formal_rl_failure_cases"])
    ).resolve()
    for path in (data_path, prediction_path, failure_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    fingerprints = [
        sha256_file(config_path),
        sha256_file(module_path),
        sha256_file(Path(__file__)),
        sha256_file(data_path),
        sha256_file(prediction_path),
        sha256_file(failure_path),
    ]
    fingerprint = hashlib.sha256("\0".join(fingerprints).encode()).hexdigest()
    timestamp = _utc_now()
    output_root = (PROJECT_ROOT / str(config["output"]["root"])).resolve()
    run_dir = output_root / "runs" / f"p0_{timestamp}_{fingerprint[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    write_json(
        {
            "status": "RUNNING",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "analysis_role": scope["analysis_role"],
            "independent_test": False,
            "plant_control_claim_authorized": False,
        },
        status_path,
    )

    encoding = str(config["output"]["encoding"])
    output_paths: list[Path] = []
    try:
        print("[p0] loading the frozen initial dataset", flush=True)
        initial = load_initial_dataset(data_path)
        feature_set = initial.feature_set
        registries = {
            target: build_paired_model_registry(
                seed=int(temporal_config["seed"]),
                n_features=len(
                    feature_set.bundles[target].feature_names[
                        str(temporal_config["feature_version"])
                    ]
                ),
                n_jobs=int(args.n_jobs),
                candidate_count=int(temporal_config["candidate_count"]),
                tabnet_config=config["tabnet"],
            )
            for target in TARGETS
        }
        outer_folds = parse_outer_folds(temporal_config["outer_folds"])
        temporal = run_temporal_validation(
            feature_set,
            registries,
            outer_folds,
            purge_days=int(temporal_config["purge_days"]),
            inner_splits=int(temporal_config["inner_splits"]),
            inner_validation_rows=int(temporal_config["inner_validation_rows"]),
            seed=int(temporal_config["seed"]),
            huber_epsilon=float(config["ensemble"]["huber_epsilon"]),
            huber_alpha=float(config["ensemble"]["huber_alpha"]),
            progress=lambda message: print(f"[p0-temporal] {message}", flush=True),
        )
        temporal_tables = {
            "temporal_predictions.csv": temporal.predictions,
            "temporal_metrics_by_fold.csv": temporal.metrics_by_fold,
            "temporal_pooled_metrics.csv": temporal.pooled_metrics,
            "temporal_selected_hyperparameters.csv": temporal.selected_hyperparameters,
            "temporal_tuning_trials.csv": temporal.tuning_trials,
            "temporal_ensemble_weights.csv": temporal.ensemble_weights,
            "temporal_outer_assignments.csv": temporal.outer_assignments,
            "temporal_inner_assignments.csv": temporal.inner_assignments,
            "temporal_leakage_audit.csv": temporal.leakage_audit,
        }
        for filename, table in temporal_tables.items():
            path = run_dir / filename
            table.to_csv(path, index=False, encoding=encoding)
            output_paths.append(path)

        print("[p0] propagating empirical proxy residual uncertainty", flush=True)
        outer_predictions = pd.read_csv(
            prediction_path, encoding="utf-8-sig", parse_dates=["Date"]
        )
        failure_cases = pd.read_csv(
            failure_path, encoding="utf-8-sig", parse_dates=["start_date", "end_date"]
        )
        residual_library = build_joint_residual_library(
            outer_predictions,
            training_window=str(uncertainty_config["training_window"]),
        )
        uncertainty = propagate_proxy_uncertainty(
            failure_cases,
            residual_library,
            correlations=tuple(float(value) for value in uncertainty_config["correlations"]),
            primary_correlation=float(uncertainty_config["primary_correlation"]),
            replicates=int(uncertainty_config["replicates"]),
            seed=int(uncertainty_config["seed"]),
            interval_level=float(uncertainty_config["interval_level"]),
        )
        uncertainty_tables = {
            "proxy_residual_library.csv": uncertainty.residual_library,
            "proxy_residual_scale.csv": uncertainty.residual_scale,
            "rl_episode_proxy_uncertainty.csv": uncertainty.episode_intervals,
            "rl_method_proxy_uncertainty_summary.csv": uncertainty.method_summary,
        }
        for filename, table in uncertainty_tables.items():
            path = run_dir / filename
            table.to_csv(path, index=False, encoding=encoding)
            output_paths.append(path)

        primary = uncertainty.method_summary.loc[
            uncertainty.method_summary["is_primary_correlation"]
        ].copy()
        evidence_summary = {
            "status": "completed",
            "analysis_role": scope["analysis_role"],
            "independent_test": False,
            "future_prediction_claim_authorized": False,
            "plant_control_claim_authorized": False,
            "temporal_validation": {
                "role": "post_hoc_rolling_origin_sensitivity_not_independent_test",
                "outer_folds": [fold.name for fold in outer_folds],
                "purge_days": int(temporal_config["purge_days"]),
                "pooled_metrics": temporal.pooled_metrics.to_dict(orient="records"),
                "model_family_selection": (
                    "frozen_final_families_only; hyperparameters and Huber meta-model "
                    "fit within each outer training period"
                ),
            },
            "proxy_uncertainty": {
                "role": (
                    "date-aggregated empirical outer-residual sensitivity; not a calibrated "
                    "counterfactual confidence interval"
                ),
                "joint_residual_dates": int(len(uncertainty.residual_library)),
                "primary_assumed_error_correlation": float(
                    uncertainty_config["primary_correlation"]
                ),
                "correlation_sensitivity_grid": list(uncertainty_config["correlations"]),
                "primary_method_summary": primary.to_dict(orient="records"),
            },
            "action_semantics": {
                "PPA": (
                    "historically field-adjustable additive; evaluated only as a virtual "
                    "counterfactual coordinate in this study"
                ),
                "DO": (
                    "measured variable with unverified measurement location; evaluated only "
                    "as a virtual sensitivity coordinate"
                ),
                "authorization": (
                    "neither variable is authorized here as a deployable control command"
                ),
            },
        }
        summary_path = run_dir / "evidence_summary.json"
        write_json(evidence_summary, summary_path)
        output_paths.append(summary_path)
    except BaseException as exc:
        write_json(
            {
                "status": "FAILED",
                "created_utc": timestamp,
                "failed_utc": _utc_now(),
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_path,
        )
        raise

    manifest = {
        "stage": "p0_evidence_boundary_audit",
        "status": "completed",
        "created_utc": timestamp,
        "fingerprint": fingerprint,
        "analysis_role": scope["analysis_role"],
        "independent_test": False,
        "future_prediction_claim_authorized": False,
        "plant_control_claim_authorized": False,
        "calibrated_counterfactual_coverage_claim_authorized": False,
        "inputs": {
            "initial_dataset": sha256_file(data_path),
            "initial_dataset_manifest": sha256_file(initial.manifest_path),
            "prediction_outer_predictions": sha256_file(prediction_path),
            "formal_rl_failure_cases": sha256_file(failure_path),
            "config": sha256_file(config_path),
            "module": sha256_file(module_path),
            "runner": sha256_file(Path(__file__)),
        },
        "outputs": {
            path.name: sha256_file(path)
            for path in sorted(output_paths, key=lambda value: value.name)
        },
    }
    manifest_path = run_dir / "manifest.json"
    write_json(manifest, manifest_path)
    latest = {
        "stage": manifest["stage"],
        "status": "completed",
        "run_dir": run_dir.relative_to(PAPER_ROOT).as_posix(),
        "manifest": manifest_path.relative_to(PAPER_ROOT).as_posix(),
        "fingerprint": fingerprint,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(latest, output_root / "LATEST.json")
    write_json(
        {
            "status": "COMPLETED",
            "created_utc": timestamp,
            "completed_utc": _utc_now(),
            "fingerprint": fingerprint,
            "manifest": "manifest.json",
        },
        status_path,
    )
    print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
