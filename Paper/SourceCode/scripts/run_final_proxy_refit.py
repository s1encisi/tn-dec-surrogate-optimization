from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.config import load_toml  # noqa: E402
from taici.final_proxy import (  # noqa: E402
    FINAL_DEC_MODEL,
    FINAL_TN_MODEL,
    artifact_contract,
    fit_final_proxy_bundle,
    freeze_candidate_table,
)
from taici.initial_dataset import load_initial_dataset  # noqa: E402
from taici.io import sha256_file, write_json  # noqa: E402
from taici.paired_random_windows import (  # noqa: E402
    BASE_MODELS,
    FEATURE_KEY,
    TARGETS,
    build_paired_model_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and refit the final ExtraTrees/Ensemble_Huber proxies."
    )
    parser.add_argument("--config", default="configs/final_workflow.toml")
    return parser.parse_args()


def _environment() -> dict[str, str]:
    packages = (
        "numpy",
        "pandas",
        "scikit-learn",
        "shap",
        "torch",
        "stable-baselines3",
        "gymnasium",
        "joblib",
        "pymoo",
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
        **versions,
    }


def _json_parameters(registries: dict[str, dict[str, Any]], lock: pd.DataFrame) -> pd.DataFrame:
    table = lock.copy()
    table["selected_parameters"] = [
        json.dumps(
            dict(registries["TN_out"][row.model].candidates[int(row.candidate_index)]),
            ensure_ascii=False,
            default=str,
        )
        for row in table.itertuples(index=False)
    ]
    return table


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_toml(str(config_path))
    paired_config_path = PROJECT_ROOT / str(config["inputs"]["paired_config"])
    paired = load_toml(str(paired_config_path))
    data_path = (PROJECT_ROOT / str(config["inputs"]["initial_dataset"])).resolve()
    initial = load_initial_dataset(data_path)
    parent_run = PROJECT_ROOT / str(config["inputs"]["parent_prediction_run"])
    unified_run = PROJECT_ROOT / str(config["inputs"]["unified_comparison_run"])
    parent_manifest_path = parent_run / "manifest.json"
    unified_manifest_path = unified_run / "manifest.json"
    parent_manifest = json.loads(parent_manifest_path.read_text(encoding="utf-8"))
    if parent_manifest["status"] != "completed":
        raise RuntimeError("The frozen parent prediction run is not completed.")
    parent_inputs = parent_manifest.get("inputs", {})
    if "initial_dataset" in parent_inputs:
        parent_input_matches = parent_inputs["initial_dataset"] == initial.file_sha256
    else:
        parent_input_matches = parent_inputs.get("data") == initial.source_sha256
    if not parent_input_matches:
        raise RuntimeError("The initial-dataset provenance differs from the parent run.")

    refit = config["refit"]
    fingerprints = (
        sha256_file(config_path),
        sha256_file(paired_config_path),
        sha256_file(data_path),
        sha256_file(parent_manifest_path),
        sha256_file(unified_manifest_path),
        sha256_file(PROJECT_ROOT / "src" / "taici" / "final_proxy.py"),
        sha256_file(Path(__file__)),
    )
    fingerprint = hashlib.sha256("\0".join(fingerprints).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = PROJECT_ROOT / str(config["output"]["root"])
    run_dir = output_root / "runs" / f"final_{timestamp}_{fingerprint[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    write_json(
        {
            "status": "RUNNING",
            "stage": "final_proxy_refit",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
        },
        status_path,
    )
    started = time.perf_counter()
    try:
        print("[final-proxy] loading the frozen paper initial dataset", flush=True)
        feature_set = initial.feature_set
        counts = pd.Series(feature_set.common_dates.year).value_counts().sort_index().to_dict()
        if counts != {2023: 362, 2024: 366, 2025: 348}:
            raise RuntimeError(f"Unexpected initial-data date counts: {counts}")
        registries = {
            target: build_paired_model_registry(
                seed=int(refit["registry_seed"]),
                n_features=len(feature_set.bundles[target].feature_names[FEATURE_KEY]),
                n_jobs=int(refit["n_jobs"]),
                candidate_count=int(paired["models"]["candidate_count"]),
                tabnet_config=paired["tabnet"],
            )
            for target in TARGETS
        }
        tuning = pd.read_csv(parent_run / "tuning_trials.csv", encoding="utf-8-sig")
        candidate_lock = freeze_candidate_table(tuning)
        candidate_lock = _json_parameters(registries, candidate_lock)
        print(
            "[final-proxy] fitting 15 TN base models with full-data three-fold OOF meta inputs",
            flush=True,
        )
        result = fit_final_proxy_bundle(
            feature_set,
            registries,
            candidate_lock,
            deployment_seed=int(refit["deployment_seed"]),
            inner_folds=int(refit["inner_folds"]),
            dec_candidate_index=int(config["final_models"]["DEC"]["candidate_index"]),
        )
        artifact_path = run_dir / "final_proxy_bundle.joblib"
        joblib.dump(result.bundle, artifact_path, compress=3)
        reloaded = joblib.load(artifact_path)
        X_tn = feature_set.bundles["TN_out"].matrix(FEATURE_KEY).iloc[:20]
        X_dec = feature_set.bundles["DEC"].matrix(FEATURE_KEY).iloc[:20]
        tn_reload_error = float(
            np.max(np.abs(result.bundle.predict_tn(X_tn) - reloaded.predict_tn(X_tn)))
        )
        dec_reload_error = float(
            np.max(np.abs(result.bundle.predict_dec(X_dec) - reloaded.predict_dec(X_dec)))
        )
        if max(tn_reload_error, dec_reload_error) > 1e-10:
            raise RuntimeError("Serialized proxy predictions do not reproduce the in-memory model.")

        selected_metrics = pd.read_csv(parent_run / "leaderboard.csv", encoding="utf-8-sig")
        evidence = selected_metrics.loc[
            (
                selected_metrics["target"].eq("DEC")
                & selected_metrics["training_window"].eq("2023_2025")
                & selected_metrics["model"].eq(FINAL_DEC_MODEL)
            )
            | (
                selected_metrics["target"].eq("TN_out")
                & selected_metrics["training_window"].eq("2023_2025")
                & selected_metrics["model"].eq(FINAL_TN_MODEL)
            )
        ].copy()
        evidence["evidence_role"] = (
            "five_overlapping_random_outer_splits_same_distribution_interpolation"
        )
        evidence["independent_test"] = False
        evidence["future_prediction_claim_authorized"] = False
        if len(evidence) != 2:
            raise RuntimeError("Could not isolate exactly two frozen performance rows.")

        dec_spec = registries["DEC"][FINAL_DEC_MODEL]
        dec_parameters = dict(
            dec_spec.candidates[int(config["final_models"]["DEC"]["candidate_index"])]
        )
        model_lock = {
            "selection_rule": (
                "user_locked_lowest_mean_RMSE_within_initial_data_2023_2025; "
                "not_unified_CRITIC_TOPSIS_unique_best"
            ),
            "DEC": {
                "model": FINAL_DEC_MODEL,
                "candidate_index": int(config["final_models"]["DEC"]["candidate_index"]),
                "parameters": dec_parameters,
            },
            "TN_out": {
                "model": FINAL_TN_MODEL,
                "base_model_order": list(BASE_MODELS),
                "candidate_lock_rule": candidate_lock.iloc[0]["selection_rule"],
                "meta_parameters": {
                    "epsilon": 1.35,
                    "alpha": 0.001,
                    "max_iter": 2000,
                    "tol": 1e-7,
                },
            },
            "outer_test_used_for_hyperparameter_lock": False,
            "full_refit_has_unseen_performance_estimate": False,
        }
        contract = artifact_contract(result.bundle)
        contract["artifact_file"] = artifact_path.name
        contract["artifact_sha256"] = sha256_file(artifact_path)
        contract["serialization_reproduction_max_abs_error"] = {
            "TN_out": tn_reload_error,
            "DEC": dec_reload_error,
        }

        encoding = str(config["output"]["encoding"])
        tables = {
            "candidate_lock.csv": candidate_lock,
            "tn_meta_oof_predictions.csv": result.oof_predictions,
            "tn_meta_oof_reconstruction_metrics.csv": result.oof_reconstruction_metrics,
            "tn_final_ensemble_weights.csv": result.ensemble_weights,
            "tn_meta_fold_assignments.csv": result.fold_assignments,
            "full_fit_predictions_diagnostic.csv": result.full_fit_predictions,
            "frozen_performance_evidence.csv": evidence,
            "feature_registry.csv": feature_set.feature_registry,
        }
        for filename, table in tables.items():
            table.to_csv(run_dir / filename, index=False, encoding=encoding)
        write_json(model_lock, run_dir / "model_selection_lock.json")
        write_json(contract, run_dir / "proxy_contract.json")
        write_json(_environment(), run_dir / "environment.json")
        elapsed = time.perf_counter() - started
        manifest = {
            "stage": "final_proxy_refit",
            "status": "completed",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "elapsed_seconds": elapsed,
            "analysis_role": config["scope"]["analysis_role"],
            "final_models": {"DEC": FINAL_DEC_MODEL, "TN_out": FINAL_TN_MODEL},
            "feature_version": FEATURE_KEY,
            "training_rows": len(feature_set.common_dates),
            "training_year_counts": counts,
            "performance_evidence_is_full_refit_score": False,
            "independent_test": False,
            "future_prediction_claim_authorized": False,
            "plant_control_claim_authorized": False,
            "inputs": {
                "initial_dataset": initial.file_sha256,
                "initial_dataset_manifest": sha256_file(initial.manifest_path),
                "initial_dataset_source": initial.source_sha256,
                "config": sha256_file(config_path),
                "paired_config": sha256_file(paired_config_path),
                "parent_manifest": sha256_file(parent_manifest_path),
                "unified_manifest": sha256_file(unified_manifest_path),
                "module": sha256_file(PROJECT_ROOT / "src" / "taici" / "final_proxy.py"),
                "runner": sha256_file(Path(__file__)),
            },
            "outputs": {
                artifact_path.name: sha256_file(artifact_path),
                **{filename: sha256_file(run_dir / filename) for filename in tables},
                "model_selection_lock.json": sha256_file(run_dir / "model_selection_lock.json"),
                "proxy_contract.json": sha256_file(run_dir / "proxy_contract.json"),
                "environment.json": sha256_file(run_dir / "environment.json"),
            },
        }
        write_json(manifest, run_dir / "manifest.json")
        write_json(
            {
                "status": "COMPLETED",
                "stage": "final_proxy_refit",
                "created_utc": timestamp,
                "completed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "manifest": "manifest.json",
            },
            status_path,
        )
        latest = {
            "stage": "final_proxy_refit",
            "status": "completed",
            "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
            "manifest": (run_dir / "manifest.json").relative_to(PROJECT_ROOT).as_posix(),
            "fingerprint": fingerprint,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(latest, output_root / "LATEST_PROXY.json")
        print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)
        return 0
    except BaseException as exc:
        write_json(
            {
                "status": "FAILED",
                "stage": "final_proxy_refit",
                "created_utc": timestamp,
                "failed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_path,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
