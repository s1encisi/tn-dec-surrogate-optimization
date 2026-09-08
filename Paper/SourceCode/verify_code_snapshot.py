"""Build CODE_MANIFEST.csv and verify copied files against their project sources."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


SNAPSHOT = Path(__file__).resolve().parent
PROJECT = SNAPSHOT.parents[1]
ARCHIVE = PROJECT / "预实验"
FINAL_RUN = ARCHIVE / "outputs" / "final_project" / "runs" / "final_20260823T080105Z_711551f0ee12"
PHASE2 = ARCHIVE / "outputs" / "phase2"
ADAPTED = {
    "configs/feature_registry.toml",
    "configs/final_workflow.toml",
    "configs/paired_random_windows.toml",
    "pyproject.toml",
    "scripts/run_final_proxy_refit.py",
    "scripts/run_final_xai.py",
    "scripts/run_paired_random_windows.py",
    "scripts/run_surrogate_rl.py",
    "src/taici/fast_proxy.py",
    "src/taici/final_proxy.py",
    "src/taici/final_xai.py",
    "src/taici/paired_random_windows.py",
    "src/taici/surrogate_rl.py",
    "tests/test_paired_random_windows.py",
    "tests/test_surrogate_rl.py",
}

GENERATED_FOR_FORMAL_RL_RERUN = {
    "configs/formal_rl_rerun.toml",
    "scripts/run_formal_rl_rerun.py",
    "tests/test_formal_rl_cuda_config.py",
    "tests/test_formal_rl_locking.py",
}

GENERATED_FOR_PAPER = {
    "configs/p0_evidence_audit.toml",
    "README_reproducibility.md",
    "scripts/run_p0_evidence_audit.py",
    "src/taici/p0_evidence.py",
    "tests/test_p0_evidence.py",
    "tests/test_configuration_contracts.py",
    "verify_code_snapshot.py",
    "tests/test_verify_code_snapshot.py",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_for(relative: Path) -> tuple[str, Path | None]:
    parts = relative.parts
    text = relative.as_posix()
    if text in GENERATED_FOR_FORMAL_RL_RERUN:
        return "generated_for_formal_rl_rerun", None
    if text in {"src/taici/initial_dataset.py", "tests/test_initial_dataset.py"}:
        return "generated_for_frozen_initial_dataset", None
    if text in ADAPTED:
        return "adapted_for_frozen_initial_dataset", ARCHIVE / relative
    if parts[0] == "plotting" or text in GENERATED_FOR_PAPER:
        return "generated_for_paper", None
    if parts[0] == "provenance":
        name = relative.name
        if name == "paper_material_qa.json":
            return "generated_for_paper", None
        if name in {"deterministic_qc_log.csv", "denoise_decision.json"}:
            return "copied_phase2_provenance", PHASE2 / name
        return "copied_final_run_provenance", FINAL_RUN / name
    if relative.as_posix() == "scripts/recompute_igd_dedup.py":
        return "copied_final_run_postprocess", FINAL_RUN / "rl" / "recompute_igd_dedup.py"
    return "copied_project_source", ARCHIVE / relative


def role(relative: Path) -> str:
    text = relative.as_posix()
    if text.startswith("configs/"):
        return "configuration"
    if text.startswith("tests/"):
        return "verification_test"
    if "phase1" in text or "phase2" in text or any(x in text for x in ("audit.py", "preprocessing.py", "denoise.py", "io.py", "config.py")):
        return "data_preprocessing"
    if "feature" in text:
        return "feature_construction"
    if any(x in text for x in ("paired_random", "raw_lag", "model_registry", "advanced_prediction", "enhanced_random", "selection", "metrics")):
        return "model_training_comparison_evaluation"
    if "final_proxy" in text or "fast_proxy" in text:
        return "final_proxy_refit"
    if "xai" in text or "explainability" in text:
        return "explainability"
    if "p0_evidence" in text:
        return "evidence_boundary_audit"
    if any(x in text for x in ("surrogate_rl", "formal_rl", "igd", "optimization_gate")):
        return "multiobjective_proxy_rl"
    if text.startswith("plotting/"):
        return "paper_figure_and_data_generation"
    if text.startswith("paper/"):
        return "paper_document_generation_and_qa"
    if text.startswith("provenance/"):
        return "frozen_provenance"
    return "project_support"


def main() -> None:
    rows = []
    failures = []
    for copied in sorted(p for p in SNAPSHOT.rglob("*") if p.is_file()):
        relative = copied.relative_to(SNAPSHOT)
        if relative.name == "CODE_MANIFEST.csv" or any(
            part in {"__pycache__", ".ruff_cache", ".pytest_cache"}
            for part in relative.parts
        ):
            continue
        category, source = source_for(relative)
        copied_hash = sha256(copied)
        source_hash = sha256(source) if source and source.exists() else ""
        if category == "adapted_for_frozen_initial_dataset":
            identical = "adapted"
        else:
            identical = "generated" if source is None else str(copied_hash == source_hash).lower()
        if source is not None and identical not in {"true", "adapted"}:
            failures.append(relative.as_posix())
        rows.append(
            {
                "category": category,
                "role": role(relative),
                "relative_path": relative.as_posix(),
                "source_path": str(source) if source else "generated within Paper/SourceCode",
                "copied_sha256": copied_hash,
                "source_sha256": source_hash,
                "identical_to_source": identical,
            }
        )
    out = SNAPSHOT / "CODE_MANIFEST.csv"
    with out.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    if failures:
        raise SystemExit(f"Snapshot mismatch: {failures}")
    print(
        f"Verified {sum(r['identical_to_source'] == 'true' for r in rows)} exact copies; "
        f"recorded {sum(r['identical_to_source'] == 'adapted' for r in rows)} adapted "
        f"files and {len(rows)} files total."
    )


if __name__ == "__main__":
    main()
