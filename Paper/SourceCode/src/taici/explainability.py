from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import itertools
from typing import Any

import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted


class ExplainabilityInputError(ValueError):
    """Raised when an explanation input violates the frozen Phase 7 contract."""


@dataclass(frozen=True)
class TransformedDesign:
    """The numeric model matrix and its auditable raw-feature lineage."""

    matrix: np.ndarray
    metadata: pd.DataFrame


@dataclass(frozen=True)
class TreeShapResult:
    """TreeSHAP values whose predictions passed an explicit additivity check."""

    values: np.ndarray
    base_values: np.ndarray
    predictions: np.ndarray
    reconstructed_predictions: np.ndarray
    transformed_design: TransformedDesign
    max_additivity_error: float


@dataclass(frozen=True)
class GroupedShapResult:
    """Local signed and global absolute SHAP after physical-group aggregation."""

    local_values: pd.DataFrame
    global_importance: pd.DataFrame
    max_conservation_error: float


@dataclass(frozen=True)
class BlockPermutationResult:
    """Repeated development-only loss increases from joint group permutations."""

    repeats: pd.DataFrame
    summary: pd.DataFrame
    baseline_rmse: float


@dataclass(frozen=True)
class ALEGrid:
    """Frozen quantile edges fitted from a named training feature only."""

    feature: str
    edges: tuple[float, ...]
    requested_bins: int
    effective_bins: int
    support_lower_quantile: float
    support_upper_quantile: float
    n_training_values: int


@dataclass(frozen=True)
class ALEResult:
    """A centered first-order ALE curve evaluated on a frozen training grid."""

    grid: ALEGrid
    curve: pd.DataFrame
    n_in_support: int
    weighted_centering_error: float


@dataclass(frozen=True)
class ALEBootstrapResult:
    """Fixed-grid circular block-bootstrap uncertainty for a first-order ALE curve."""

    point: ALEResult
    samples: pd.DataFrame
    interval: pd.DataFrame
    requested_replicates: int
    attempted_resamples: int


@dataclass(frozen=True)
class FoldStabilityResult:
    """Feature-level fold stability and all pairwise fold rank correlations."""

    feature_summary: pd.DataFrame
    pairwise_spearman: pd.DataFrame
    median_pairwise_spearman: float


def validate_development_only(
    frame: pd.DataFrame,
    *,
    development_start: str | pd.Timestamp,
    development_end: str | pd.Timestamp,
    expected_target: str = "TN_out",
    date_column: str = "Date",
    partition_column: str = "study_partition",
    target_column: str | None = None,
) -> pd.DataFrame:
    """Return a sorted copy after enforcing the Phase 7 development boundary.

    The function deliberately accepts a development subset; it does not require
    the first and last study dates to be present. Any duplicate date, mixed target,
    unauthorized partition, or row outside the inclusive development interval is
    rejected before an explanation can be computed.
    """

    if frame.empty:
        raise ExplainabilityInputError("Explainability input cannot be empty.")
    required = {date_column, partition_column}
    missing = required.difference(frame.columns)
    if missing:
        raise ExplainabilityInputError(f"Missing development-scope columns: {sorted(missing)}")
    result = frame.copy()
    try:
        result[date_column] = pd.to_datetime(result[date_column], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ExplainabilityInputError("Explanation dates must be valid timestamps.") from exc
    if result[date_column].isna().any():
        raise ExplainabilityInputError("Explanation dates cannot be missing.")
    if result[date_column].duplicated().any():
        raise ExplainabilityInputError("Explanation dates must be unique.")
    partitions = set(result[partition_column].dropna().astype(str))
    if result[partition_column].isna().any() or partitions != {"development"}:
        raise ExplainabilityInputError(
            "Phase 7 may consume development rows only; fixed-test rows are forbidden."
        )
    lower = pd.Timestamp(development_start)
    upper = pd.Timestamp(development_end)
    if lower > upper:
        raise ExplainabilityInputError("development_start must not follow development_end.")
    if result[date_column].lt(lower).any() or result[date_column].gt(upper).any():
        raise ExplainabilityInputError(
            "Explanation dates fall outside the frozen development interval."
        )
    if target_column is not None:
        if target_column not in result:
            raise ExplainabilityInputError(f"Missing target identity column: {target_column}")
        targets = set(result[target_column].dropna().astype(str))
        if result[target_column].isna().any() or targets != {expected_target}:
            raise ExplainabilityInputError(
                f"Phase 7 is authorized for {expected_target} only; observed {sorted(targets)}."
            )
    return result.sort_values(date_column, kind="stable").reset_index(drop=True)


def validate_phase7_scope(
    frame: pd.DataFrame,
    *,
    target: str,
    model_name: str,
    development_start: str | pd.Timestamp,
    development_end: str | pd.Timestamp,
) -> pd.DataFrame:
    """Enforce the frozen full-XAI admission: TN_out development ExtraTrees only."""

    if target != "TN_out":
        raise ExplainabilityInputError("Full Phase 7 XAI is authorized for TN_out only.")
    if model_name != "ExtraTrees":
        raise ExplainabilityInputError(
            "The frozen Phase 7 primary explanation model is ExtraTrees."
        )
    return validate_development_only(
        frame,
        development_start=development_start,
        development_end=development_end,
    )


def _feature_group_lookup(
    feature_names: Sequence[str],
    group_definitions: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    names = tuple(str(name) for name in feature_names)
    if not names or len(set(names)) != len(names):
        raise ExplainabilityInputError("Raw feature names must be non-empty and unique.")
    known = set(names)
    lookup: dict[str, str] = {}
    for group, members in group_definitions.items():
        group_name = str(group)
        if not group_name:
            raise ExplainabilityInputError("Feature group names cannot be empty.")
        for feature in (str(member) for member in members):
            if feature not in known:
                raise ExplainabilityInputError(
                    f"Feature group {group_name} references unknown feature {feature}."
                )
            if feature in lookup:
                raise ExplainabilityInputError(
                    f"Raw feature {feature} is assigned to more than one physical group."
                )
            lookup[feature] = group_name
    unassigned = [name for name in names if name not in lookup]
    if unassigned:
        raise ExplainabilityInputError(
            f"Every raw feature must have one physical group; unassigned: {unassigned}"
        )
    return lookup


def pipeline_feature_mapping(
    pipeline: Pipeline,
    feature_names: Sequence[str],
    group_definitions: Mapping[str, Sequence[str]],
) -> pd.DataFrame:
    """Map fitted Pipeline columns, including missing indicators, to raw groups."""

    if not isinstance(pipeline, Pipeline) or "imputer" not in pipeline.named_steps:
        raise ExplainabilityInputError("A fitted Pipeline with an 'imputer' step is required.")
    if len(pipeline.steps) < 2:
        raise ExplainabilityInputError("The Pipeline must include preprocessing and a model.")
    names = tuple(str(name) for name in feature_names)
    lookup = _feature_group_lookup(names, group_definitions)
    imputer = pipeline.named_steps["imputer"]
    try:
        check_is_fitted(imputer)
        transformed_names = tuple(str(name) for name in imputer.get_feature_names_out(names))
    except (AttributeError, ValueError) as exc:
        raise ExplainabilityInputError(
            "The Pipeline imputer must be fitted on the supplied raw feature schema."
        ) from exc

    indicator_indices: tuple[int, ...] = ()
    indicator = getattr(imputer, "indicator_", None)
    if indicator is not None:
        indicator_indices = tuple(int(index) for index in indicator.features_)
    expected_sources = [*names, *(names[index] for index in indicator_indices)]
    if len(transformed_names) != len(expected_sources):
        raise ExplainabilityInputError(
            "Cannot reconcile transformed Pipeline columns with imputer feature lineage."
        )
    records = []
    for index, (transformed, source) in enumerate(
        zip(transformed_names, expected_sources, strict=True)
    ):
        is_indicator = index >= len(names)
        records.append(
            {
                "transformed_index": index,
                "transformed_feature": transformed,
                "source_feature": source,
                "physical_group": lookup[source],
                "is_missing_indicator": is_indicator,
            }
        )
    return pd.DataFrame.from_records(records)


def transform_pipeline_design(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    group_definitions: Mapping[str, Sequence[str]],
) -> TransformedDesign:
    """Transform raw columns while retaining a checked one-to-one column audit."""

    names = tuple(str(name) for name in feature_names)
    missing = set(names).difference(X.columns)
    if missing:
        raise ExplainabilityInputError(f"Raw explanation matrix is missing: {sorted(missing)}")
    metadata = pipeline_feature_mapping(pipeline, names, group_definitions)
    try:
        transformed = pipeline[:-1].transform(X.loc[:, list(names)])
    except (TypeError, ValueError) as exc:
        raise ExplainabilityInputError("Pipeline preprocessing failed for explanation data.") from exc
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    matrix = np.asarray(transformed, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(metadata):
        raise ExplainabilityInputError(
            "Transformed model matrix and audited feature mapping have different dimensions."
        )
    if not np.isfinite(matrix).all():
        raise ExplainabilityInputError("Pipeline preprocessing produced NaN or infinite values.")
    return TransformedDesign(matrix=matrix, metadata=metadata)


def compute_tree_shap(
    pipeline: Pipeline,
    X: pd.DataFrame,
    feature_names: Sequence[str],
    group_definitions: Mapping[str, Sequence[str]],
    *,
    additivity_tolerance: float = 1e-8,
) -> TreeShapResult:
    """Compute TreeSHAP on transformed columns and fail if additivity is violated."""

    if not np.isfinite(additivity_tolerance) or additivity_tolerance <= 0:
        raise ExplainabilityInputError("additivity_tolerance must be positive and finite.")
    design = transform_pipeline_design(pipeline, X, feature_names, group_definitions)
    model = pipeline.steps[-1][1]
    try:
        check_is_fitted(model)
        explanation = shap.TreeExplainer(model)(design.matrix, check_additivity=False)
    except Exception as exc:
        raise ExplainabilityInputError(
            "The fitted final estimator is not compatible with TreeSHAP."
        ) from exc
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[..., 0]
    if values.ndim != 2 or values.shape != design.matrix.shape:
        raise ExplainabilityInputError(
            f"Expected one SHAP value per transformed feature; observed {values.shape}."
        )
    base = np.asarray(explanation.base_values, dtype=float)
    if base.size == 1:
        base_values = np.repeat(float(base.reshape(-1)[0]), len(values))
    else:
        base_values = base.reshape(len(values), -1)
        if base_values.shape[1] != 1:
            raise ExplainabilityInputError("Only single-output tree regression is supported.")
        base_values = base_values[:, 0]
    predictions = np.asarray(model.predict(design.matrix), dtype=float).reshape(-1)
    reconstructed = base_values + values.sum(axis=1)
    if not (
        np.isfinite(values).all()
        and np.isfinite(base_values).all()
        and np.isfinite(predictions).all()
    ):
        raise ExplainabilityInputError("TreeSHAP returned NaN or infinite values.")
    maximum_error = float(np.max(np.abs(predictions - reconstructed), initial=0.0))
    if maximum_error > additivity_tolerance:
        raise ExplainabilityInputError(
            "TreeSHAP additivity check failed: "
            f"maximum error {maximum_error:.3g} exceeds {additivity_tolerance:.3g}."
        )
    return TreeShapResult(
        values=values,
        base_values=base_values,
        predictions=predictions,
        reconstructed_predictions=reconstructed,
        transformed_design=design,
        max_additivity_error=maximum_error,
    )


def aggregate_grouped_shap(
    shap_values: np.ndarray,
    feature_metadata: pd.DataFrame,
    *,
    conservation_tolerance: float = 1e-10,
) -> GroupedShapResult:
    """Sum local signed SHAP first, then compute absolute group importance."""

    values = np.asarray(shap_values, dtype=float)
    required = {"transformed_index", "physical_group"}
    if values.ndim != 2 or not required.issubset(feature_metadata.columns):
        raise ExplainabilityInputError("SHAP values or transformed feature metadata are invalid.")
    metadata = feature_metadata.sort_values("transformed_index", kind="stable")
    if len(metadata) != values.shape[1] or metadata["transformed_index"].tolist() != list(
        range(values.shape[1])
    ):
        raise ExplainabilityInputError("SHAP columns are not aligned with feature metadata.")
    groups = metadata["physical_group"].astype(str).drop_duplicates().tolist()
    local = pd.DataFrame(index=pd.RangeIndex(len(values)))
    for group in groups:
        indices = metadata.loc[
            metadata["physical_group"].astype(str).eq(group), "transformed_index"
        ].to_numpy(dtype=int)
        local[group] = values[:, indices].sum(axis=1)
    error = float(np.max(np.abs(local.sum(axis=1).to_numpy() - values.sum(axis=1)), initial=0.0))
    if error > conservation_tolerance:
        raise ExplainabilityInputError("Grouped SHAP does not conserve local signed contributions.")
    importance = pd.DataFrame(
        {
            "physical_group": groups,
            "mean_abs_group_shap": [float(local[group].abs().mean()) for group in groups],
            "mean_signed_group_shap": [float(local[group].mean()) for group in groups],
        }
    )
    total = float(importance["mean_abs_group_shap"].sum())
    importance["share_of_grouped_mean_abs"] = (
        importance["mean_abs_group_shap"] / total if total > 0 else 0.0
    )
    importance = importance.sort_values(
        ["mean_abs_group_shap", "physical_group"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)
    return GroupedShapResult(
        local_values=local,
        global_importance=importance,
        max_conservation_error=error,
    )


def block_permutation_indices(
    n_rows: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute complete contiguous blocks; keep a short terminal remainder fixed."""

    if isinstance(n_rows, bool) or not isinstance(n_rows, int) or n_rows < 1:
        raise ExplainabilityInputError("n_rows must be a positive integer.")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ExplainabilityInputError("block_length must be a positive integer.")
    n_complete_blocks = n_rows // block_length
    if n_complete_blocks < 2:
        raise ExplainabilityInputError("At least two complete blocks are required.")
    complete = np.arange(n_complete_blocks * block_length).reshape(
        n_complete_blocks, block_length
    )
    order = rng.permutation(n_complete_blocks)
    permuted = complete[order].reshape(-1)
    remainder = np.arange(n_complete_blocks * block_length, n_rows)
    return np.concatenate([permuted, remainder]).astype(int, copy=False)


def joint_group_block_permutation(
    estimator: Any,
    X: pd.DataFrame,
    y: Sequence[float] | np.ndarray,
    dates: Sequence[Any] | pd.Series,
    group_definitions: Mapping[str, Sequence[str]],
    *,
    development_start: str | pd.Timestamp,
    development_end: str | pd.Timestamp,
    block_length: int = 7,
    repeats: int = 100,
    random_seed: int = 20260816,
    study_partition: Sequence[str] | pd.Series | None = None,
) -> BlockPermutationResult:
    """Jointly permute each physical group's raw columns in contiguous time blocks."""

    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ExplainabilityInputError("repeats must be a positive integer.")
    if X.empty or len(X) != len(y) or len(X) != len(dates):
        raise ExplainabilityInputError("X, y, and dates must be non-empty and aligned.")
    scope = pd.DataFrame(
        {
            "Date": pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="raise"),
            "study_partition": (
                pd.Series(study_partition).reset_index(drop=True)
                if study_partition is not None
                else "development"
            ),
        }
    )
    validated = validate_development_only(
        scope,
        development_start=development_start,
        development_end=development_end,
    )
    if not validated["Date"].equals(scope["Date"]):
        raise ExplainabilityInputError("Permutation rows must already be in chronological order.")
    feature_names = tuple(str(name) for name in X.columns)
    _feature_group_lookup(feature_names, group_definitions)
    target = np.asarray(y, dtype=float).reshape(-1)
    if not np.isfinite(target).all():
        raise ExplainabilityInputError("Permutation outcomes must be finite and cannot be imputed.")
    baseline_prediction = np.asarray(estimator.predict(X), dtype=float).reshape(-1)
    if len(baseline_prediction) != len(target) or not np.isfinite(baseline_prediction).all():
        raise ExplainabilityInputError("The fitted estimator returned invalid predictions.")
    baseline_rmse = float(np.sqrt(np.mean(np.square(target - baseline_prediction))))
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, Any]] = []
    for group, members in group_definitions.items():
        columns = [str(member) for member in members]
        for repeat in range(repeats):
            indices = block_permutation_indices(len(X), block_length, rng)
            permuted = X.copy()
            # One shared index vector is essential: all variables in a physical
            # group move together, retaining their multivariate block structure.
            permuted.loc[:, columns] = X.iloc[indices][columns].to_numpy()
            prediction = np.asarray(estimator.predict(permuted), dtype=float).reshape(-1)
            permuted_rmse = float(np.sqrt(np.mean(np.square(target - prediction))))
            records.append(
                {
                    "physical_group": str(group),
                    "repeat": repeat,
                    "baseline_RMSE": baseline_rmse,
                    "permuted_RMSE": permuted_rmse,
                    "delta_RMSE": permuted_rmse - baseline_rmse,
                    "block_length": block_length,
                    "random_seed": random_seed,
                    "study_partition": "development",
                    "fixed_test_accessed": False,
                }
            )
    repeat_frame = pd.DataFrame.from_records(records)
    summary = (
        repeat_frame.groupby("physical_group", sort=False)["delta_RMSE"]
        .agg(
            mean_delta_RMSE="mean",
            median_delta_RMSE="median",
            ci_low=lambda values: float(np.quantile(values, 0.025)),
            ci_high=lambda values: float(np.quantile(values, 0.975)),
        )
        .reset_index()
        .sort_values(
            ["mean_delta_RMSE", "physical_group"],
            ascending=[False, True],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    return BlockPermutationResult(
        repeats=repeat_frame,
        summary=summary,
        baseline_rmse=baseline_rmse,
    )


def fit_ale_grid(
    training_X: pd.DataFrame,
    feature: str,
    *,
    quantile_bins: int = 10,
    support_lower_quantile: float = 0.05,
    support_upper_quantile: float = 0.95,
) -> ALEGrid:
    """Fit and freeze unique ALE bin edges using training observations only."""

    if feature not in training_X:
        raise ExplainabilityInputError(f"ALE training matrix lacks feature {feature}.")
    if isinstance(quantile_bins, bool) or not isinstance(quantile_bins, int) or quantile_bins < 2:
        raise ExplainabilityInputError("quantile_bins must be an integer of at least two.")
    if not (0 <= support_lower_quantile < support_upper_quantile <= 1):
        raise ExplainabilityInputError("ALE support quantiles must satisfy 0 <= low < high <= 1.")
    values = pd.to_numeric(training_X[feature], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < max(10, quantile_bins):
        raise ExplainabilityInputError("Too few finite training values to fit the ALE grid.")
    probabilities = np.linspace(
        support_lower_quantile, support_upper_quantile, quantile_bins + 1
    )
    edges = np.unique(np.quantile(values, probabilities))
    if len(edges) < 3:
        raise ExplainabilityInputError("ALE feature has insufficient unique training support.")
    return ALEGrid(
        feature=feature,
        edges=tuple(float(value) for value in edges),
        requested_bins=quantile_bins,
        effective_bins=len(edges) - 1,
        support_lower_quantile=float(support_lower_quantile),
        support_upper_quantile=float(support_upper_quantile),
        n_training_values=int(len(values)),
    )


def compute_ale_1d(
    estimator: Any,
    X: pd.DataFrame,
    grid: ALEGrid,
) -> ALEResult:
    """Compute centered first-order ALE without refitting or changing frozen edges."""

    if grid.feature not in X:
        raise ExplainabilityInputError(f"ALE evaluation matrix lacks feature {grid.feature}.")
    edges = np.asarray(grid.edges, dtype=float)
    if len(edges) < 3 or not np.all(np.diff(edges) > 0):
        raise ExplainabilityInputError("ALE grid edges must be finite and strictly increasing.")
    values = pd.to_numeric(X[grid.feature], errors="coerce").to_numpy(dtype=float)
    in_support = np.isfinite(values) & (values >= edges[0]) & (values <= edges[-1])
    if not in_support.any():
        raise ExplainabilityInputError("No ALE evaluation rows lie inside training support.")
    supported_X = X.loc[in_support].reset_index(drop=True)
    supported_values = values[in_support]
    bin_index = np.searchsorted(edges, supported_values, side="right") - 1
    bin_index = np.clip(bin_index, 0, len(edges) - 2)
    local_effects: list[float] = []
    counts: list[int] = []
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        selected = np.flatnonzero(bin_index == index)
        if len(selected) == 0:
            raise ExplainabilityInputError(
                "An ALE bin is empty on the evaluation sample; evaluate on the training support "
                "or reduce the number of bins."
            )
        lower_frame = supported_X.iloc[selected].copy()
        upper_frame = supported_X.iloc[selected].copy()
        lower_frame.loc[:, grid.feature] = lower
        upper_frame.loc[:, grid.feature] = upper
        lower_prediction = np.asarray(estimator.predict(lower_frame), dtype=float).reshape(-1)
        upper_prediction = np.asarray(estimator.predict(upper_frame), dtype=float).reshape(-1)
        differences = upper_prediction - lower_prediction
        if len(differences) != len(selected) or not np.isfinite(differences).all():
            raise ExplainabilityInputError("ALE intervention predictions are invalid.")
        local_effects.append(float(np.mean(differences)))
        counts.append(int(len(selected)))
    effects = np.asarray(local_effects)
    uncentered = np.cumsum(effects) - 0.5 * effects
    weights = np.asarray(counts, dtype=float)
    centered = uncentered - float(np.average(uncentered, weights=weights))
    centering_error = float(abs(np.average(centered, weights=weights)))
    curve = pd.DataFrame(
        {
            "feature": grid.feature,
            "bin": np.arange(len(effects), dtype=int),
            "lower": edges[:-1],
            "upper": edges[1:],
            "center": (edges[:-1] + edges[1:]) / 2.0,
            "n": counts,
            "mean_local_effect": effects,
            "ALE": centered,
            "within_training_support": True,
        }
    )
    return ALEResult(
        grid=grid,
        curve=curve,
        n_in_support=int(in_support.sum()),
        weighted_centering_error=centering_error,
    )


def _circular_block_bootstrap_indices(
    n_rows: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n_rows < 2 or block_length < 1:
        raise ExplainabilityInputError(
            "ALE block bootstrap requires at least two rows and a positive block length."
        )
    n_blocks = int(np.ceil(n_rows / block_length))
    starts = rng.integers(0, n_rows, size=n_blocks)
    sampled = np.concatenate(
        [(start + np.arange(block_length)) % n_rows for start in starts]
    )
    return sampled[:n_rows].astype(int, copy=False)


def block_bootstrap_ale_1d(
    estimator: Any,
    X: pd.DataFrame,
    grid: ALEGrid,
    dates: Sequence[Any] | pd.Series,
    *,
    development_start: str | pd.Timestamp,
    development_end: str | pd.Timestamp,
    replicates: int = 500,
    block_length: int = 7,
    random_seed: int = 20260816,
    study_partition: Sequence[str] | pd.Series | None = None,
) -> ALEBootstrapResult:
    """Estimate ALE sampling bands using a frozen grid and circular time blocks.

    The fitted estimator and training-derived ``grid`` remain unchanged. Resampling
    therefore quantifies empirical curve stability, not model-refit uncertainty. A
    resample missing an ALE bin is retried rather than silently interpolated.
    """

    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ExplainabilityInputError("replicates must be a positive integer.")
    if isinstance(block_length, bool) or not isinstance(block_length, int) or block_length < 1:
        raise ExplainabilityInputError("block_length must be a positive integer.")
    if X.empty or len(X) != len(dates):
        raise ExplainabilityInputError("ALE X and dates must be non-empty and aligned.")
    scope = pd.DataFrame(
        {
            "Date": pd.to_datetime(pd.Series(dates).reset_index(drop=True), errors="raise"),
            "study_partition": (
                pd.Series(study_partition).reset_index(drop=True)
                if study_partition is not None
                else "development"
            ),
        }
    )
    validated = validate_development_only(
        scope,
        development_start=development_start,
        development_end=development_end,
    )
    if not validated["Date"].equals(scope["Date"]):
        raise ExplainabilityInputError("ALE bootstrap rows must be chronologically ordered.")

    point = compute_ale_1d(estimator, X, grid)
    rng = np.random.default_rng(random_seed)
    sample_records: list[dict[str, Any]] = []
    successful = 0
    attempts = 0
    maximum_attempts = max(replicates * 25, 100)
    while successful < replicates and attempts < maximum_attempts:
        attempts += 1
        indices = _circular_block_bootstrap_indices(len(X), block_length, rng)
        try:
            sampled_curve = compute_ale_1d(
                estimator, X.iloc[indices].reset_index(drop=True), grid
            ).curve
        except ExplainabilityInputError as exc:
            if "ALE bin is empty" in str(exc):
                continue
            raise
        for row in sampled_curve.itertuples(index=False):
            sample_records.append(
                {
                    "replicate": successful,
                    "bin": int(row.bin),
                    "center": float(row.center),
                    "ALE": float(row.ALE),
                    "block_length": block_length,
                    "study_partition": "development",
                    "fixed_test_accessed": False,
                }
            )
        successful += 1
    if successful != replicates:
        raise ExplainabilityInputError(
            "Unable to obtain the requested ALE bootstrap replicates without empty bins."
        )
    samples = pd.DataFrame.from_records(sample_records)
    interval = (
        samples.groupby(["bin", "center"], sort=True)["ALE"]
        .agg(
            bootstrap_median="median",
            ci_low=lambda values: float(np.quantile(values, 0.025)),
            ci_high=lambda values: float(np.quantile(values, 0.975)),
        )
        .reset_index()
        .merge(point.curve[["bin", "ALE"]].rename(columns={"ALE": "point_ALE"}), on="bin")
    )
    interval["replicates"] = replicates
    return ALEBootstrapResult(
        point=point,
        samples=samples,
        interval=interval,
        requested_replicates=replicates,
        attempted_resamples=attempts,
    )


def summarize_fold_stability(
    importances: pd.DataFrame,
    *,
    fold_column: str = "fold",
    feature_column: str = "feature",
    importance_column: str = "mean_abs_shap",
    signed_column: str = "mean_signed_shap",
    top_k: int = 5,
) -> FoldStabilityResult:
    """Summarize deterministic top-k and direction stability across time folds."""

    required = {fold_column, feature_column, importance_column, signed_column}
    missing = required.difference(importances.columns)
    if missing:
        raise ExplainabilityInputError(f"Fold stability input is missing: {sorted(missing)}")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ExplainabilityInputError("top_k must be a positive integer.")
    data = importances.loc[:, list(required)].copy()
    data[importance_column] = pd.to_numeric(data[importance_column], errors="coerce")
    data[signed_column] = pd.to_numeric(data[signed_column], errors="coerce")
    if (
        data[[fold_column, feature_column]].isna().any().any()
        or not np.isfinite(data[[importance_column, signed_column]].to_numpy(float)).all()
        or data[importance_column].lt(0).any()
    ):
        raise ExplainabilityInputError("Fold stability values must be complete and finite.")
    if data.duplicated([fold_column, feature_column]).any():
        raise ExplainabilityInputError("Each fold/feature pair must occur exactly once.")
    feature_sets = data.groupby(fold_column, sort=False)[feature_column].agg(
        lambda values: frozenset(values.astype(str))
    )
    if len(feature_sets) < 2 or feature_sets.nunique() != 1:
        raise ExplainabilityInputError("At least two folds with identical feature sets are required.")
    n_features = len(feature_sets.iloc[0])
    if top_k > n_features:
        raise ExplainabilityInputError("top_k cannot exceed the number of features.")
    data[feature_column] = data[feature_column].astype(str)
    ranked_parts = []
    for _, fold in data.groupby(fold_column, sort=False):
        ranked = fold.sort_values(
            [importance_column, feature_column], ascending=[False, True], kind="stable"
        ).copy()
        ranked["ordinal_rank"] = np.arange(1, len(ranked) + 1)
        ranked["in_top_k"] = ranked["ordinal_rank"].le(top_k)
        ranked_parts.append(ranked)
    ranked_data = pd.concat(ranked_parts, ignore_index=True)

    summary_records = []
    for feature, subset in ranked_data.groupby(feature_column, sort=True):
        signs = np.sign(subset[signed_column].to_numpy(float))
        counts = {value: int(np.sum(signs == value)) for value in (-1.0, 0.0, 1.0)}
        dominant_sign = max(counts, key=lambda value: (counts[value], value != 0, value))
        direction = {1.0: "positive", -1.0: "negative", 0.0: "zero"}[dominant_sign]
        summary_records.append(
            {
                "feature": feature,
                "median_mean_abs_shap": float(subset[importance_column].median()),
                "importance_iqr": float(
                    subset[importance_column].quantile(0.75)
                    - subset[importance_column].quantile(0.25)
                ),
                "median_rank": float(subset["ordinal_rank"].median()),
                "top_k_frequency": float(subset["in_top_k"].mean()),
                "dominant_direction": direction,
                "direction_consistency": float(counts[dominant_sign] / len(subset)),
                "n_folds": int(len(subset)),
            }
        )
    feature_summary = pd.DataFrame.from_records(summary_records).sort_values(
        ["median_mean_abs_shap", "feature"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)

    pivot = ranked_data.pivot(
        index=feature_column, columns=fold_column, values="ordinal_rank"
    )
    pairwise_records = []
    for first, second in itertools.combinations(pivot.columns, 2):
        first_values = pivot[first].to_numpy(float)
        second_values = pivot[second].to_numpy(float)
        correlation = float(spearmanr(first_values, second_values).statistic)
        if not np.isfinite(correlation):
            correlation = 1.0 if np.array_equal(first_values, second_values) else 0.0
        pairwise_records.append(
            {"fold_a": str(first), "fold_b": str(second), "spearman_rho": correlation}
        )
    pairwise = pd.DataFrame.from_records(pairwise_records)
    median = float(pairwise["spearman_rho"].median())
    return FoldStabilityResult(
        feature_summary=feature_summary,
        pairwise_spearman=pairwise,
        median_pairwise_spearman=median,
    )


def select_preregistered_local_cases(
    predictions: pd.DataFrame,
    *,
    development_start: str | pd.Timestamp,
    development_end: str | pd.Timestamp,
    middle_quantile_low: float = 0.40,
    middle_quantile_high: float = 0.60,
    case_types: Sequence[str] = (
        "typical_accurate",
        "highest_target",
        "largest_error",
    ),
) -> pd.DataFrame:
    """Select local cases by fixed rules, with earliest-date tie breaking."""

    required = {"Date", "actual", "prediction", "study_partition"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ExplainabilityInputError(f"Local-case input is missing: {sorted(missing)}")
    authorized = ("typical_accurate", "highest_target", "largest_error")
    requested = tuple(str(case) for case in case_types)
    if len(set(requested)) != len(requested) or not requested:
        raise ExplainabilityInputError("Local case types must be non-empty and unique.")
    unknown = set(requested).difference(authorized)
    if unknown:
        raise ExplainabilityInputError(f"Unknown local case rules: {sorted(unknown)}")
    if not (0 <= middle_quantile_low < middle_quantile_high <= 1):
        raise ExplainabilityInputError("Middle quantiles must satisfy 0 <= low < high <= 1.")
    data = validate_development_only(
        predictions,
        development_start=development_start,
        development_end=development_end,
    )
    data["actual"] = pd.to_numeric(data["actual"], errors="coerce")
    data["prediction"] = pd.to_numeric(data["prediction"], errors="coerce")
    if not np.isfinite(data[["actual", "prediction"]].to_numpy(float)).all():
        raise ExplainabilityInputError("Local-case truth and predictions must be finite.")
    data["absolute_error"] = (data["actual"] - data["prediction"]).abs()
    low = float(data["actual"].quantile(middle_quantile_low))
    high = float(data["actual"].quantile(middle_quantile_high))
    middle = data.loc[data["actual"].between(low, high, inclusive="both")]
    if middle.empty:
        raise ExplainabilityInputError("No observations satisfy the typical-case quantile rule.")
    choices = {
        "typical_accurate": middle.sort_values(
            ["absolute_error", "Date"], ascending=[True, True], kind="stable"
        ).iloc[0],
        "highest_target": data.sort_values(
            ["actual", "Date"], ascending=[False, True], kind="stable"
        ).iloc[0],
        "largest_error": data.sort_values(
            ["absolute_error", "Date"], ascending=[False, True], kind="stable"
        ).iloc[0],
    }
    records = []
    for case in requested:
        row = choices[case]
        records.append(
            {
                "case_type": case,
                "Date": pd.Timestamp(row["Date"]),
                "actual": float(row["actual"]),
                "prediction": float(row["prediction"]),
                "absolute_error": float(row["absolute_error"]),
                "selection_rule": {
                    "typical_accurate": (
                        f"minimum absolute error among actual quantiles "
                        f"[{middle_quantile_low:.2f}, {middle_quantile_high:.2f}]"
                    ),
                    "highest_target": "maximum observed target",
                    "largest_error": "maximum absolute prediction error",
                }[case],
                "tie_break": "earliest_date",
                "study_partition": "development",
                "fixed_test_accessed": False,
            }
        )
    return pd.DataFrame.from_records(records)
