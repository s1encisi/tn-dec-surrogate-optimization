"""Numerically equivalent fast inference for the frozen TN/DEC proxies.

The reinforcement-learning environment needs many single-row predictions.
Scikit-learn forest estimators have substantial Python/joblib overhead in that
regime, so this module flattens the already-fitted tree structures and traverses
the same thresholds with Numba. No component, coefficient, or tree is removed.
The runner must pass the full-panel fidelity audit before using this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from numba import njit
import numpy as np
import pandas as pd

from .final_proxy import FinalProxyBundle, FinalProxyError, align_feature_frame


FAST_COMPONENTS = ("RandomForest", "ExtraTrees", "AdaBoost")


class FastProxyError(ValueError):
    """Raised when exact fast inference cannot preserve the frozen proxy."""


@njit(cache=False)
def _predict_forest_mean(
    values: np.ndarray,
    roots: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    split_features: np.ndarray,
    thresholds: np.ndarray,
    leaf_values: np.ndarray,
) -> np.ndarray:
    prediction = np.empty(values.shape[0], dtype=np.float64)
    for sample in range(values.shape[0]):
        total = 0.0
        for tree_index in range(roots.shape[0]):
            node = roots[tree_index]
            while children_left[node] != -1:
                if values[sample, split_features[node]] <= thresholds[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            total += leaf_values[node]
        prediction[sample] = total / roots.shape[0]
    return prediction


@njit(cache=False)
def _predict_weighted_median(
    values: np.ndarray,
    roots: np.ndarray,
    children_left: np.ndarray,
    children_right: np.ndarray,
    split_features: np.ndarray,
    thresholds: np.ndarray,
    leaf_values: np.ndarray,
    estimator_weights: np.ndarray,
) -> np.ndarray:
    prediction = np.empty(values.shape[0], dtype=np.float64)
    half_weight = 0.5 * np.sum(estimator_weights)
    for sample in range(values.shape[0]):
        tree_predictions = np.empty(roots.shape[0], dtype=np.float64)
        for tree_index in range(roots.shape[0]):
            node = roots[tree_index]
            while children_left[node] != -1:
                if values[sample, split_features[node]] <= thresholds[node]:
                    node = children_left[node]
                else:
                    node = children_right[node]
            tree_predictions[tree_index] = leaf_values[node]
        order = np.argsort(tree_predictions)
        cumulative_weight = 0.0
        selected = order[-1]
        for order_position in range(order.shape[0]):
            selected = order[order_position]
            cumulative_weight += estimator_weights[selected]
            if cumulative_weight >= half_weight:
                break
        prediction[sample] = tree_predictions[selected]
    return prediction


@dataclass(frozen=True)
class _FlattenedTreeEnsemble:
    imputer: Any
    roots: np.ndarray
    children_left: np.ndarray
    children_right: np.ndarray
    split_features: np.ndarray
    thresholds: np.ndarray
    leaf_values: np.ndarray
    estimator_weights: np.ndarray | None

    @classmethod
    def from_pipeline(
        cls, pipeline: Any, *, weighted_median: bool = False
    ) -> "_FlattenedTreeEnsemble":
        if not hasattr(pipeline, "named_steps"):
            raise FastProxyError("A fast tree component must be a fitted Pipeline.")
        if "imputer" not in pipeline.named_steps or "model" not in pipeline.named_steps:
            raise FastProxyError("The fitted tree Pipeline lacks imputer/model steps.")
        model = pipeline.named_steps["model"]
        estimators = tuple(np.asarray(model.estimators_, dtype=object).reshape(-1))
        if not estimators:
            raise FastProxyError("The fitted tree ensemble contains no estimators.")
        roots: list[int] = []
        left_parts: list[np.ndarray] = []
        right_parts: list[np.ndarray] = []
        feature_parts: list[np.ndarray] = []
        threshold_parts: list[np.ndarray] = []
        value_parts: list[np.ndarray] = []
        offset = 0
        for estimator in estimators:
            tree = estimator.tree_
            roots.append(offset)
            left = tree.children_left.astype(np.int64).copy()
            right = tree.children_right.astype(np.int64).copy()
            left[left >= 0] += offset
            right[right >= 0] += offset
            left_parts.append(left)
            right_parts.append(right)
            feature_parts.append(tree.feature.astype(np.int64))
            threshold_parts.append(tree.threshold.astype(np.float64))
            value_parts.append(tree.value[:, 0, 0].astype(np.float64))
            offset += int(tree.node_count)
        weights: np.ndarray | None = None
        if weighted_median:
            weights = np.asarray(model.estimator_weights_, dtype=np.float64).reshape(-1)
            if len(weights) != len(estimators) or not np.isfinite(weights).all():
                raise FastProxyError("AdaBoost estimator weights are invalid.")
        return cls(
            imputer=pipeline.named_steps["imputer"],
            roots=np.asarray(roots, dtype=np.int64),
            children_left=np.concatenate(left_parts),
            children_right=np.concatenate(right_parts),
            split_features=np.concatenate(feature_parts),
            thresholds=np.concatenate(threshold_parts),
            leaf_values=np.concatenate(value_parts),
            estimator_weights=weights,
        )

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        values = np.asarray(self.imputer.transform(frame), dtype=np.float32)
        arguments = (
            values,
            self.roots,
            self.children_left,
            self.children_right,
            self.split_features,
            self.thresholds,
            self.leaf_values,
        )
        if self.estimator_weights is None:
            return _predict_forest_mean(*arguments)
        return _predict_weighted_median(*arguments, self.estimator_weights)


@dataclass
class FastExactProxyBundle:
    """Drop-in exact-inference interface with compiled tree traversal."""

    original: FinalProxyBundle
    tn_fast_components: Mapping[str, _FlattenedTreeEnsemble]
    dec_fast_model: _FlattenedTreeEnsemble

    @classmethod
    def from_frozen(cls, bundle: FinalProxyBundle) -> "FastExactProxyBundle":
        components = bundle.tn_model.base_models
        missing = set(FAST_COMPONENTS).difference(components)
        if missing:
            raise FastProxyError(f"Frozen TN ensemble lacks components: {sorted(missing)}")
        fast_components = {
            "RandomForest": _FlattenedTreeEnsemble.from_pipeline(
                components["RandomForest"]
            ),
            "ExtraTrees": _FlattenedTreeEnsemble.from_pipeline(components["ExtraTrees"]),
            "AdaBoost": _FlattenedTreeEnsemble.from_pipeline(
                components["AdaBoost"], weighted_median=True
            ),
        }
        return cls(
            original=bundle,
            tn_fast_components=fast_components,
            dec_fast_model=_FlattenedTreeEnsemble.from_pipeline(bundle.dec_model),
        )

    @property
    def tn_feature_names(self) -> tuple[str, ...]:
        return self.original.tn_feature_names

    @property
    def dec_feature_names(self) -> tuple[str, ...]:
        return self.original.dec_feature_names

    @property
    def deployment_seed(self) -> int:
        return self.original.deployment_seed

    @property
    def training_dates(self) -> tuple[str, ...]:
        return self.original.training_dates

    def predict_dec(self, X: pd.DataFrame) -> np.ndarray:
        frame = align_feature_frame(X, self.dec_feature_names)
        prediction = self.dec_fast_model.predict(frame)
        if not np.isfinite(prediction).all():
            raise FinalProxyError("Fast DEC ExtraTrees returned invalid predictions.")
        return prediction

    def predict_tn(self, X: pd.DataFrame) -> np.ndarray:
        frame = align_feature_frame(X, self.tn_feature_names)
        component_values: list[np.ndarray] = []
        for name in self.original.tn_model.component_order:
            if name in self.tn_fast_components:
                prediction = self.tn_fast_components[name].predict(frame)
            else:
                prediction = np.asarray(
                    self.original.tn_model.base_models[name].predict(frame), dtype=float
                ).reshape(-1)
            if len(prediction) != len(frame) or not np.isfinite(prediction).all():
                raise FinalProxyError(f"Fast TN component {name} returned invalid predictions.")
            component_values.append(prediction)
        matrix = np.column_stack(component_values)
        prediction = (
            self.original.tn_model.effective_intercept
            + matrix @ self.original.tn_model.effective_weights
        )
        if not np.isfinite(prediction).all():
            raise FinalProxyError("Fast TN Ensemble_Huber returned invalid predictions.")
        return np.asarray(prediction, dtype=float)


def audit_fast_proxy_fidelity(
    original: FinalProxyBundle,
    fast: FastExactProxyBundle,
    X_tn: pd.DataFrame,
    X_dec: pd.DataFrame,
    *,
    tolerance: float = 1e-9,
) -> dict[str, float | int | bool | str]:
    """Require full-panel numerical equivalence before accelerated RL calls."""

    tn_error = float(
        np.max(np.abs(original.predict_tn(X_tn) - fast.predict_tn(X_tn)))
    )
    dec_error = float(
        np.max(np.abs(original.predict_dec(X_dec) - fast.predict_dec(X_dec)))
    )
    passed = bool(max(tn_error, dec_error) <= float(tolerance))
    result: dict[str, float | int | bool | str] = {
        "status": "PASSED" if passed else "FAILED",
        "rows_TN": int(len(X_tn)),
        "rows_DEC": int(len(X_dec)),
        "max_abs_error_TN": tn_error,
        "max_abs_error_DEC": dec_error,
        "absolute_tolerance": float(tolerance),
        "components_preserved": len(original.tn_model.component_order),
        "acceleration_role": (
            "same_fitted_trees_and_all_15_Huber_components_not_distillation"
        ),
    }
    if not passed:
        raise FastProxyError(f"Fast proxy fidelity failed: {result}")
    return result
