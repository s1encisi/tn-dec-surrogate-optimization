from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any
import warnings

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from taici.config import load_toml  # noqa: E402
from taici.final_proxy import FinalProxyBundle  # noqa: E402
from taici.final_xai import run_crossfitted_xai, run_deployment_xai  # noqa: E402
from taici.initial_dataset import load_initial_dataset  # noqa: E402
from taici.io import sha256_file, write_json  # noqa: E402
from taici.paired_random_windows import (  # noqa: E402
    FEATURE_KEY,
    TARGETS,
    build_paired_model_registry,
)


BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREEN = "#009E73"
PURPLE = "#CC79A7"
BLACK = "#000000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run final proxy XAI for DEC and TN_out.")
    parser.add_argument("--config", default="configs/final_workflow.toml")
    parser.add_argument("--proxy-run-dir", default=None)
    return parser.parse_args()


def _resolve_proxy_run(config: dict[str, Any], argument: str | None) -> Path:
    if argument:
        path = Path(argument)
        return path if path.is_absolute() else PROJECT_ROOT / path
    latest_path = PROJECT_ROOT / str(config["output"]["root"]) / "LATEST_PROXY.json"
    latest = json.loads(latest_path.read_text(encoding="utf-8"))
    if latest.get("status") != "completed":
        raise RuntimeError("The latest final proxy run is not completed.")
    return PROJECT_ROOT / str(latest["run_dir"])


def _importance_ale_figure(
    permutation: pd.DataFrame, ale: pd.DataFrame, output_base: Path
) -> list[Path]:
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 7.4), layout="constrained")
    target_settings = {
        "TN_out": (BLUE, "TN_out", "mg/L"),
        "DEC": (VERMILLION, "DEC", "kWh/d"),
    }
    for row, target in enumerate(("TN_out", "DEC")):
        color, label, unit = target_settings[target]
        importance = permutation.loc[permutation["target"].eq(target)].sort_values(
            "mean_delta_RMSE", ascending=True
        )
        ax = axes[row, 0]
        values = importance["mean_delta_RMSE"].to_numpy(float)
        low = importance["q025_delta_RMSE"].to_numpy(float)
        high = importance["q975_delta_RMSE"].to_numpy(float)
        errors = np.vstack([np.maximum(values - low, 0), np.maximum(high - values, 0)])
        ax.barh(
            importance["feature"],
            values,
            xerr=errors,
            color=color,
            edgecolor=BLACK,
            linewidth=0.5,
            alpha=0.86,
        )
        ax.axvline(0, color=BLACK, linewidth=0.8)
        ax.set_xlabel(f"Cross-fitted permutation ΔRMSE ({unit})")
        ax.set_title(f"{chr(65 + row * 3)}  {label}: held-out importance", loc="left")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.5)
        for column, feature, feature_label in (
            (1, "PPA", "PPA"),
            (2, "DO", "DO"),
        ):
            ax = axes[row, column]
            curve = ale.loc[ale["target"].eq(target) & ale["feature"].eq(feature)]
            ax.plot(
                curve["center"],
                curve["ALE"],
                color=color,
                marker="o" if column == 1 else "s",
                linewidth=1.8,
                markersize=4,
            )
            ax.axhline(0, color=BLACK, linewidth=0.8, linestyle="--")
            ax.set_xlabel(feature_label)
            ax.set_ylabel(f"ALE ({unit})")
            ax.set_title(
                f"{chr(65 + row * 3 + column)}  {label}: {feature_label}", loc="left"
            )
            ax.grid(color="#E5E5E5", linewidth=0.5)
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, facecolor="white")
    fig.savefig(pdf, facecolor="white")
    plt.close(fig)
    return [png, pdf]


def _shap_explanation(table: pd.DataFrame, feature_order: list[str], keys: list[str]) -> Any:
    values = table.pivot(index=keys, columns="feature", values="shap_value").loc[
        :, feature_order
    ]
    data = table.pivot(index=keys, columns="feature", values="feature_value").loc[
        values.index, feature_order
    ]
    base = table.groupby(keys, observed=True)["base_value"].first().loc[values.index]
    return shap.Explanation(
        values=values.to_numpy(float),
        base_values=base.to_numpy(float),
        data=data.to_numpy(float),
        feature_names=feature_order,
    )


def _beeswarm_figure(
    table: pd.DataFrame,
    feature_order: list[str],
    keys: list[str],
    title: str,
    x_label: str,
    output_base: Path,
) -> list[Path]:
    explanation = _shap_explanation(table, feature_order, keys)
    ax = shap.plots.beeswarm(
        explanation,
        max_display=len(feature_order),
        alpha=0.65,
        s=18,
        group_remaining_features=False,
        show=False,
    )
    ax.set_title(title, loc="left")
    ax.set_xlabel(x_label)
    ax.figure.set_size_inches(8.2, 5.6)
    ax.figure.tight_layout()
    png = output_base.with_suffix(".png")
    pdf = output_base.with_suffix(".pdf")
    ax.figure.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    ax.figure.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(ax.figure)
    return [png, pdf]


def main() -> int:
    args = parse_args()
    config_path = PROJECT_ROOT / args.config
    config = load_toml(str(config_path))
    xai_config = config["xai"]
    paired_config_path = PROJECT_ROOT / str(config["inputs"]["paired_config"])
    paired = load_toml(str(paired_config_path))
    data_path = (PROJECT_ROOT / str(config["inputs"]["initial_dataset"])).resolve()
    initial = load_initial_dataset(data_path)
    parent_run = PROJECT_ROOT / str(config["inputs"]["parent_prediction_run"])
    proxy_run = _resolve_proxy_run(config, args.proxy_run_dir)
    proxy_manifest = json.loads((proxy_run / "manifest.json").read_text(encoding="utf-8"))
    if proxy_manifest.get("status") != "completed":
        raise RuntimeError("The selected final proxy run is not completed.")
    final_bundle = joblib.load(proxy_run / "final_proxy_bundle.joblib")
    if not isinstance(final_bundle, FinalProxyBundle):
        raise RuntimeError("The serialized final proxy has an unexpected type.")

    fingerprints = (
        sha256_file(config_path),
        sha256_file(data_path),
        sha256_file(proxy_run / "manifest.json"),
        sha256_file(proxy_run / "final_proxy_bundle.joblib"),
        sha256_file(parent_run / "manifest.json"),
        sha256_file(PROJECT_ROOT / "src" / "taici" / "final_xai.py"),
        sha256_file(Path(__file__)),
    )
    fingerprint = hashlib.sha256("\0".join(fingerprints).encode()).hexdigest()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = proxy_run / "xai"
    output_dir.mkdir(parents=True, exist_ok=False)
    status_path = output_dir / "run_status.json"
    write_json(
        {
            "status": "RUNNING",
            "stage": "final_xai",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
        },
        status_path,
    )
    started = time.perf_counter()
    try:
        warnings.filterwarnings(
            "ignore",
            message=(
                r"^X does not have valid feature names, but LGBMRegressor was fitted "
                r"with feature names$"
            ),
            category=UserWarning,
        )
        feature_set = initial.feature_set
        refit = config["refit"]
        # The frozen parent experiment used run_paired_random_windows.py's
        # default n_jobs=-1. XGBoost's histogram builder is sensitive to the
        # thread count, so outer-model reconstruction must preserve it exactly.
        parent_reconstruction_n_jobs = -1
        registries = {
            target: build_paired_model_registry(
                seed=int(refit["registry_seed"]),
                n_features=len(feature_set.bundles[target].feature_names[FEATURE_KEY]),
                n_jobs=parent_reconstruction_n_jobs,
                candidate_count=int(paired["models"]["candidate_count"]),
                tabnet_config=paired["tabnet"],
            )
            for target in TARGETS
        }
        print("[final-xai] reconstructing five outer models and held-out explanations", flush=True)
        crossfitted = run_crossfitted_xai(
            feature_set,
            registries,
            parent_run,
            permutation_repeats=int(xai_config["permutation_repeats"]),
            random_seed=int(xai_config["random_seed"]),
        )
        print("[final-xai] computing full-refit TN permutation-SHAP and ALE", flush=True)
        deployment = run_deployment_xai(
            final_bundle,
            feature_set,
            crossfitted,
            parent_run,
            tn_shap_rows=int(xai_config["tn_shap_rows"]),
            tn_background_rows=int(xai_config["tn_shap_background_rows"]),
            tn_shap_permutations=int(xai_config["tn_shap_permutations"]),
            tn_shap_additivity_tolerance=float(
                xai_config["tn_shap_additivity_tolerance_mg_L"]
            ),
            ale_features=tuple(str(value) for value in xai_config["ale_features"]),
            ale_quantile_bins=int(xai_config["ale_quantile_bins"]),
            ale_support_lower_quantile=float(xai_config["ale_support_lower_quantile"]),
            ale_support_upper_quantile=float(xai_config["ale_support_upper_quantile"]),
            random_seed=int(xai_config["random_seed"]),
        )
        encoding = str(config["output"]["encoding"])
        tables = {
            "outer_reproduction_audit.csv": crossfitted.reproduction_audit,
            "permutation_importance_repeats.csv": crossfitted.permutation_repeats,
            "permutation_importance_summary.csv": crossfitted.permutation_summary,
            "dec_crossfitted_shap_values.csv": crossfitted.dec_shap_values,
            "dec_shap_audit.csv": crossfitted.dec_shap_audit,
            "dec_shap_global.csv": deployment.dec_shap_global,
            "tn_deployment_shap_values.csv": deployment.tn_shap_values,
            "tn_shap_audit.csv": deployment.tn_shap_audit,
            "tn_shap_global.csv": deployment.tn_shap_global,
            "ale_curves.csv": deployment.ale_curves,
            "local_cases.csv": deployment.local_cases,
        }
        for filename, table in tables.items():
            table.to_csv(output_dir / filename, index=False, encoding=encoding)
        figure_paths = _importance_ale_figure(
            crossfitted.permutation_summary,
            deployment.ale_curves,
            output_dir / "Fig_XAI_overview",
        )
        dec_order = deployment.dec_shap_global["feature"].tolist()
        tn_order = deployment.tn_shap_global["feature"].tolist()
        figure_paths.extend(
            _beeswarm_figure(
                crossfitted.dec_shap_values,
                dec_order,
                ["seed", "Date"],
                "DEC ExtraTrees: cross-fitted TreeSHAP",
                "SHAP value (kWh/d)",
                output_dir / "Fig_XAI_DEC_SHAP",
            )
        )
        figure_paths.extend(
            _beeswarm_figure(
                deployment.tn_shap_values,
                tn_order,
                ["Date"],
                "TN_out Ensemble_Huber: model-agnostic permutation SHAP",
                "SHAP value (mg/L)",
                output_dir / "Fig_XAI_TN_SHAP",
            )
        )
        figure_manifest = {
            "general_figure_standard": "provisional_publication_ready_no_target_journal_claim",
            "palette": "Okabe-Ito on-white subset with non-color panel separation",
            "figures": {
                "Fig_XAI_overview": {
                    "caption": (
                        "Cross-fitted permutation importance and full-refit ALE within the "
                        "5th-95th percentile empirical support. Error bars are descriptive "
                        "2.5th-97.5th percentiles across five overlapping random holdouts and "
                        "permutations, not confidence intervals."
                    ),
                    "alt_text": (
                        "Two-row scientific figure. TN and DEC feature permutation importance "
                        "are shown beside PPA and DO ALE curves."
                    ),
                },
                "Fig_XAI_DEC_SHAP": {
                    "caption": "TreeSHAP for 350 cross-fitted DEC outer-holdout explanations.",
                    "alt_text": "Beeswarm of signed DEC TreeSHAP values by model input.",
                },
                "Fig_XAI_TN_SHAP": {
                    "caption": (
                        "Permutation SHAP for deterministic representative rows of the full-data "
                        "TN Ensemble_Huber artifact using a shared training background."
                    ),
                    "alt_text": "Beeswarm of signed TN Ensemble_Huber SHAP values by model input.",
                },
            },
            "causal_claim_authorized": False,
        }
        write_json(figure_manifest, output_dir / "figure_manifest.json")
        elapsed = time.perf_counter() - started
        manifest = {
            "stage": "final_xai",
            "status": "completed",
            "created_utc": timestamp,
            "fingerprint": fingerprint,
            "elapsed_seconds": elapsed,
            "targets": {"DEC": "ExtraTrees", "TN_out": "Ensemble_Huber"},
            "feature_version": FEATURE_KEY,
            "old_phase7_reused_as_scientific_result": False,
            "dec_scope": "five_crossfitted_random_outer_holdouts",
            "tn_global_scope": "full_data_refit_representative_rows",
            "permutation_scope": "five_crossfitted_random_outer_holdouts",
            "parent_reconstruction_n_jobs": parent_reconstruction_n_jobs,
            "tn_shap_additivity_tolerance_mg_L": float(
                xai_config["tn_shap_additivity_tolerance_mg_L"]
            ),
            "tn_shap_additivity_tolerance_role": (
                "fixed_numerical_roundoff_guard_for_15_model_permutation_explainer"
            ),
            "independent_test": False,
            "future_prediction_claim_authorized": False,
            "causal_claim_authorized": False,
            "inputs": {
                "config": sha256_file(config_path),
                "initial_dataset": initial.file_sha256,
                "initial_dataset_manifest": sha256_file(initial.manifest_path),
                "initial_dataset_source": initial.source_sha256,
                "proxy_manifest": sha256_file(proxy_run / "manifest.json"),
                "proxy_artifact": sha256_file(proxy_run / "final_proxy_bundle.joblib"),
                "parent_prediction_manifest": sha256_file(parent_run / "manifest.json"),
                "module": sha256_file(PROJECT_ROOT / "src" / "taici" / "final_xai.py"),
                "runner": sha256_file(Path(__file__)),
            },
            "outputs": {
                **{filename: sha256_file(output_dir / filename) for filename in tables},
                **{path.name: sha256_file(path) for path in figure_paths},
                "figure_manifest.json": sha256_file(output_dir / "figure_manifest.json"),
            },
        }
        write_json(manifest, output_dir / "manifest.json")
        write_json(
            {
                "status": "COMPLETED",
                "stage": "final_xai",
                "created_utc": timestamp,
                "completed_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "fingerprint": fingerprint,
                "manifest": "manifest.json",
            },
            status_path,
        )
        latest_path = PROJECT_ROOT / str(config["output"]["root"]) / "LATEST_XAI.json"
        write_json(
            {
                "stage": "final_xai",
                "status": "completed",
                "run_dir": proxy_run.relative_to(PROJECT_ROOT).as_posix(),
                "xai_dir": output_dir.relative_to(PROJECT_ROOT).as_posix(),
                "manifest": (output_dir / "manifest.json").relative_to(PROJECT_ROOT).as_posix(),
                "fingerprint": fingerprint,
            },
            latest_path,
        )
        print(json.dumps({"status": "completed", "xai_dir": str(output_dir)}, indent=2))
        return 0
    except BaseException as exc:
        write_json(
            {
                "status": "FAILED",
                "stage": "final_xai",
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
