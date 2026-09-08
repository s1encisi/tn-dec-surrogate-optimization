from __future__ import annotations

import argparse
from collections.abc import Mapping
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.config import load_toml  # noqa: E402
from taici.initial_dataset import (  # noqa: E402
    DEC_FEATURES,
    TN_FEATURES,
    load_initial_dataset,
)
from taici.io import sha256_file, write_json  # noqa: E402
from taici.paired_random_windows import (  # noqa: E402
    BASE_MODELS,
    FEATURE_KEY,
    OUTER_SEEDS,
    TARGETS,
    build_paired_model_registry,
    make_shared_2025_assignments,
    run_paired_random_windows,
)


EXPECTED_REGISTRY_SCHEMA = "formal_initial_dataset_feature_registry_v1"
EXPECTED_FIELD_NAMESPACE = "Paper/InitialData/initial_dataset.csv"
HISTORICAL_ALIAS_PREFIX = "past3" "_mean_"
EXPECTED_FORMAL_FEATURES = {
    "TN_out": TN_FEATURES,
    "DEC": DEC_FEATURES,
}


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} must be a TOML table.")
    return value


def _legacy_token_locations(value: object, path: tuple[str, ...] = ()) -> list[str]:
    locations: list[str] = []
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key)
            nested_path = (*path, key)
            if HISTORICAL_ALIAS_PREFIX in key:
                locations.append(".".join(nested_path))
            locations.extend(_legacy_token_locations(nested, nested_path))
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            locations.extend(_legacy_token_locations(nested, (*path, str(index))))
    elif isinstance(value, str) and HISTORICAL_ALIAS_PREFIX in value:
        locations.append(".".join(path))
    return locations


def _validate_feature_registry(
    feature_config: Mapping[str, Any],
    paired_config: Mapping[str, Any],
    initial_manifest: Mapping[str, Any],
) -> None:
    """Fail closed unless all formal feature declarations share one namespace."""

    expected_top_level = {"registry", "targets", "historical_aliases"}
    if set(feature_config) != expected_top_level:
        raise RuntimeError(
            "Feature registry top-level tables differ from the formal contract: "
            f"observed={sorted(feature_config)}, expected={sorted(expected_top_level)}."
        )

    registry = _mapping(feature_config["registry"], "feature registry metadata")
    expected_registry_fields = {
        "schema_version",
        "feature_version",
        "field_namespace",
        "targets",
        "target_history_used",
        "other_effluent_features_used",
        "availability_status",
    }
    if set(registry) != expected_registry_fields:
        raise RuntimeError("Feature registry metadata fields differ from the formal contract.")
    if registry["schema_version"] != EXPECTED_REGISTRY_SCHEMA:
        raise RuntimeError("Feature registry schema version is not recognized.")
    if registry["feature_version"] != FEATURE_KEY:
        raise RuntimeError("Feature registry version differs from the frozen feature key.")
    if registry["field_namespace"] != EXPECTED_FIELD_NAMESPACE:
        raise RuntimeError("Feature registry field namespace is not the formal initial dataset.")
    if tuple(registry["targets"]) != TARGETS:
        raise RuntimeError("Feature registry targets differ from the frozen paired protocol.")
    if bool(registry["target_history_used"]):
        raise RuntimeError("Formal feature registry must prohibit target history.")
    if bool(registry["other_effluent_features_used"]):
        raise RuntimeError("Formal feature registry must prohibit other effluent features.")

    if tuple(initial_manifest.get("target_columns", ())) != TARGETS:
        raise RuntimeError("Initial-dataset manifest targets differ from the paired protocol.")
    if initial_manifest.get("feature_version") != FEATURE_KEY:
        raise RuntimeError("Initial-dataset manifest feature version differs from the registry.")

    targets = _mapping(feature_config["targets"], "feature registry targets")
    if tuple(targets) != TARGETS:
        raise RuntimeError("Feature registry must contain TN_out and DEC in frozen order only.")
    paired_features = _mapping(paired_config.get("features"), "paired feature configuration")
    expected_paired_fields = {"version", "TN_out_predictors", "DEC_predictors"}
    if set(paired_features) != expected_paired_fields:
        raise RuntimeError("Paired feature configuration fields differ from the formal contract.")
    if paired_features["version"] != FEATURE_KEY:
        raise RuntimeError("Paired feature version differs from the frozen feature key.")

    manifest_keys = {"TN_out": "tn_predictors", "DEC": "dec_predictors"}
    paired_keys = {"TN_out": "TN_out_predictors", "DEC": "DEC_predictors"}
    for target, expected in EXPECTED_FORMAL_FEATURES.items():
        spec = _mapping(targets.get(target), f"feature registry target {target}")
        if set(spec) != {"predictors", "expected_dimension"}:
            raise RuntimeError(f"Feature registry fields for {target} differ from the contract.")
        configured = tuple(str(value) for value in spec["predictors"])
        if configured != expected:
            raise RuntimeError(f"Feature registry predictors for {target} differ from code.")
        if int(spec["expected_dimension"]) != len(expected):
            raise RuntimeError(f"Feature registry dimension for {target} is inconsistent.")
        if tuple(initial_manifest.get(manifest_keys[target], ())) != expected:
            raise RuntimeError(f"Initial-dataset manifest predictors for {target} differ from code.")
        if tuple(paired_features[paired_keys[target]]) != expected:
            raise RuntimeError(f"Paired-config predictors for {target} differ from code.")

    aliases = _mapping(feature_config["historical_aliases"], "historical aliases")
    provenance_mapping = _mapping(
        initial_manifest.get("provenance_column_mapping"),
        "initial-dataset provenance mapping",
    )
    expected_aliases = _mapping(
        provenance_mapping.get("historical_to_formal"),
        "initial-dataset historical-to-formal mapping",
    )
    if dict(aliases) != dict(expected_aliases):
        raise RuntimeError("Historical aliases differ from the frozen initial-dataset manifest.")

    misplaced = [
        location
        for location in _legacy_token_locations(feature_config)
        if not location.startswith("historical_aliases.")
    ]
    if misplaced:
        raise RuntimeError(
            "Legacy historical aliases may appear only under historical_aliases: "
            + ", ".join(misplaced)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired post-hoc random comparison: 2025-only versus 2023-2025."
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / "configs" / "paired_random_windows.toml"
    feature_config_path = PROJECT_ROOT / "configs" / "feature_registry.toml"
    config = load_toml(str(config_path))
    feature_config = load_toml(str(feature_config_path))
    data_path = (PROJECT_ROOT / str(config["input"]["initial_dataset"])).resolve()
    module_path = PROJECT_ROOT / "src" / "taici" / "paired_random_windows.py"
    protocol = config["protocol"]
    if tuple(protocol["targets"]) != TARGETS:
        raise RuntimeError("Configured targets differ from the frozen paired protocol.")
    if tuple(int(seed) for seed in protocol["outer_seeds"]) != OUTER_SEEDS:
        raise RuntimeError("Configured seeds differ from the frozen paired protocol.")
    if tuple(config["models"]["base_pool"]) != BASE_MODELS:
        raise RuntimeError("Configured base pool differs from the frozen 15-model registry.")

    print("[paired-random] loading the frozen paper initial dataset", flush=True)
    initial = load_initial_dataset(data_path)
    _validate_feature_registry(feature_config, config, initial.manifest)
    feature_set = initial.feature_set
    year_counts = pd.Series(feature_set.common_dates.year).value_counts().sort_index().to_dict()
    expected_counts = {2023: 362, 2024: 366, 2025: 348}
    if year_counts != expected_counts:
        raise RuntimeError(f"Unexpected initial-data counts: {year_counts}; expected {expected_counts}.")
    assignments = make_shared_2025_assignments(
        feature_set.common_dates,
        seeds=tuple(int(value) for value in protocol["outer_seeds"]),
        test_fraction=float(protocol["outer_test_fraction"]),
    )
    role_counts = assignments.groupby(["seed", "role"], observed=True).size().unstack()
    if not role_counts["outer_test"].eq(70).all() or not role_counts["outer_train"].eq(278).all():
        raise RuntimeError(f"Unexpected shared 2025 assignment counts:\n{role_counts}")

    fingerprints = (
        sha256_file(config_path),
        sha256_file(feature_config_path),
        sha256_file(data_path),
        sha256_file(module_path),
        sha256_file(Path(__file__)),
    )
    fingerprint = hashlib.sha256("\0".join(fingerprints).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_root = PROJECT_ROOT / str(config["output"]["root"])
    run_dir = output_root / "runs" / f"paired_{timestamp}_{fingerprint[:12]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "run_status.json"
    write_json(
        {
            "status": "RUNNING",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "post_hoc": True,
            "independent_test": False,
        },
        status_path,
    )
    print(
        "[paired-random] common dates="
        f"{year_counts}; per seed: test=70, train2025=278, train3y=1006",
        flush=True,
    )
    print(f"[paired-random] output={run_dir}", flush=True)
    try:
        registries = {
            target: build_paired_model_registry(
                seed=int(protocol["outer_seeds"][0]),
                n_features=len(feature_set.bundles[target].feature_names[config["features"]["version"]]),
                n_jobs=int(args.n_jobs),
                candidate_count=int(config["models"]["candidate_count"]),
                tabnet_config=config["tabnet"],
            )
            for target in TARGETS
        }
        result = run_paired_random_windows(
            feature_set,
            registries,
            assignments,
            inner_folds=int(protocol["inner_folds"]),
            ensemble_model_ids=tuple(config["ensemble"]["eligible_base_models"]),
            top_k_candidates=tuple(
                int(value) for value in config["ensemble"]["mean_top_k_candidates"]
            ),
            ridge_alphas=tuple(float(value) for value in config["ensemble"]["ridge_alphas"]),
            bootstrap_replicates=int(config["bootstrap"]["replicates"]),
            bootstrap_seed=int(config["bootstrap"]["seed"]),
            progress=lambda message: print(f"[paired-random] {message}", flush=True),
        )
    except BaseException as exc:
        write_json(
            {
                "status": "FAILED",
                "created_utc": timestamp,
                "failed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "error": f"{type(exc).__name__}: {exc}",
            },
            status_path,
        )
        raise

    tables = {
        "outer_predictions.csv": result.outer_predictions,
        "inner_oof_predictions.csv": result.inner_oof_predictions,
        "metrics_by_seed.csv": result.metrics_by_seed,
        "leaderboard.csv": result.leaderboard,
        "window_comparison_by_seed.csv": result.window_comparison_by_seed,
        "window_comparison_summary.csv": result.window_comparison_summary,
        "window_comparison_daily.csv": result.window_comparison_daily,
        "tuning_trials.csv": result.tuning_trials,
        "selected_hyperparameters.csv": result.selected_hyperparameters,
        "ensemble_weights.csv": result.ensemble_weights,
        "outer_assignments.csv": result.outer_assignments,
        "inner_assignments.csv": result.inner_assignments,
        "failures.csv": result.failures,
        "leakage_audit.csv": result.leakage_audit,
        "completeness_audit.csv": result.completeness_audit,
        "feature_registry.csv": feature_set.feature_registry,
    }
    encoding = str(config["output"]["encoding"])
    for filename, table in tables.items():
        table.to_csv(run_dir / filename, index=False, encoding=encoding)
    manifest = {
        "stage": "paired_random_2025_vs_2023_2025_nested_oof",
        "status": "completed",
        "created_utc": timestamp,
        "fingerprint": fingerprint,
        "analysis_role": protocol["analysis_role"],
        "ordinary_daily_random": True,
        "outer_split": "shared_2025_ordinary_row_random_80_20_five_seeds",
        "outer_seeds": list(protocol["outer_seeds"]),
        "test_rows_per_seed": 70,
        "train_rows_2025_only_per_seed": 278,
        "train_rows_2023_2025_per_seed": 1006,
        "feature_version": config["features"]["version"],
        "absolute_date_anchor": protocol["absolute_date_anchor"],
        "base_models": list(BASE_MODELS),
        "baseline": ["TrainingMean"],
        "ensembles": list(config["ensemble"]["methods"]),
        "target_history_used": False,
        "other_effluent_features_used": False,
        "outer_test_accessed_for_tuning_or_weights": False,
        "test_period_previously_seen": True,
        "independent_test": False,
        "future_prediction_claim_authorized": False,
        "all_model_results_retained": True,
        "post_hoc_worst_comparator_is_not_a_formal_baseline": True,
        "window_interval_role": (
            "descriptive seed-resampling only; the five overlapping random splits are not "
            "independent and the interval is not a significance test"
        ),
        "inputs": {
            "initial_dataset": initial.file_sha256,
            "initial_dataset_manifest": sha256_file(initial.manifest_path),
            "initial_dataset_source": initial.source_sha256,
            "config": sha256_file(config_path),
            "feature_registry_config": sha256_file(feature_config_path),
            "module": sha256_file(module_path),
            "runner": sha256_file(Path(__file__)),
        },
        "outputs": {filename: sha256_file(run_dir / filename) for filename in tables},
    }
    write_json(manifest, run_dir / "manifest.json")
    latest = {
        "stage": manifest["stage"],
        "status": "completed",
        "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
        "manifest": (run_dir / "manifest.json").relative_to(PROJECT_ROOT).as_posix(),
        "fingerprint": fingerprint,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(latest, output_root / "LATEST.json")
    write_json(
        {
            "status": "COMPLETED",
            "created_utc": timestamp,
            "completed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "fingerprint": fingerprint,
            "manifest": "manifest.json",
        },
        status_path,
    )
    print(json.dumps(latest, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
