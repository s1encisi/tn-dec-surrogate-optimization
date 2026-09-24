"""Invariant tests for the public synthetic demonstration.

The suite targets the guarantees the demonstration actually claims: sealed-test
isolation, training-only preprocessing, exact optimiser budgets, bounded candidate
search and hash-consistent evidence artifacts.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from tn_dec_demo.workflow import (
    SEARCH_METHODS,
    Settings,
    feature_names,
    fit_select,
    nondominated,
    registry,
    run,
    same_domain_solutions,
    search_method,
    sha,
    splits,
    synthetic_data,
)

TRAINING_MEAN = "TrainingMean"


class LinearToyModel:
    """Minimal stand-in for a fitted surrogate, so search logic can be tested alone."""

    def __init__(self, target: str) -> None:
        self.target = target

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.target == "TN_out":
            return frame.PPA.to_numpy() + 0.1 * frame.DO.to_numpy()
        return 2 - frame.PPA.to_numpy() + 0.1 * frame.DO.to_numpy()


class SignedToyModel:
    """Surrogate whose two objectives move in opposite directions."""

    def __init__(self, sign: int) -> None:
        self.sign = sign

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.sign * frame.PPA.to_numpy() + frame.DO.to_numpy()


class DemoTests(unittest.TestCase):
    """Behavioural tests for the synthetic demonstration workflow."""

    def test_synthetic_fixture_reproducibility_and_missingness(self) -> None:
        first = synthetic_data(180, 17)
        pd.testing.assert_frame_equal(first, synthetic_data(180, 17))
        self.assertFalse(first.equals(synthetic_data(180, 18)))
        self.assertTrue(first["DO"].isna().any())
        self.assertFalse(first[["TN_out", "DEC"]].isna().any().any())
        self.assertEqual(str(first.Date.iloc[0].date()), "2000-01-04")

    def test_split_is_a_disjoint_complete_partition(self) -> None:
        parts = splits(480, 17)
        sets = [set(values) for values in parts.values()]
        self.assertEqual([len(part) for part in sets], [288, 96, 96])
        self.assertEqual(set.union(*sets), set(range(480)))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_test_data_cannot_change_selection_or_fitted_models(self) -> None:
        frame = synthetic_data(180, 17)
        parts = splits(len(frame), 17)
        first, models, _, trials = fit_select(frame, parts, 17)

        changed = frame.copy()
        changed.loc[parts["test"], feature_names("TN_out")] = 1e7
        changed.loc[parts["test"], ["TN_out", "DEC"]] = -1e8
        second, models_from_changed, _, _ = fit_select(changed, parts, 17)

        self.assertEqual(first, second)
        for target in first:
            columns = feature_names(target)
            validation = frame.loc[parts["validation"], columns]
            np.testing.assert_allclose(
                models[target].predict(validation),
                models_from_changed[target].predict(validation),
                atol=1e-9,
                rtol=1e-9,
            )
            candidates = [
                row for row in trials if row["target"] == target and row["model"] != TRAINING_MEAN
            ]
            best = min(candidates, key=lambda row: (row["RMSE"], row["model"]))["model"]
            self.assertEqual(first[target], best)

    def test_preprocessing_statistics_are_fitted_on_training_rows(self) -> None:
        frame = synthetic_data(180, 29)
        parts = splits(len(frame), 29)
        columns = feature_names("TN_out")
        estimator = registry(29)["RidgePolynomial"].fit(
            frame.loc[parts["train"], columns], frame.loc[parts["train"], "TN_out"]
        )
        actual = estimator.named_steps["simpleimputer"].statistics_
        expected = frame.loc[parts["train"], columns].median().to_numpy()
        np.testing.assert_allclose(actual, expected)

    def test_pareto_archive_handles_duplicates_and_dominance(self) -> None:
        values = np.array([[0, 1], [1, 0], [0.5, 0.5], [0.75, 0.75], [0.5, 0.5]])
        self.assertEqual(set(nondominated(values)), {0, 1, 2})
        with self.assertRaises(ValueError):
            nondominated(np.array([[float("nan"), 1]]))

    def test_each_search_method_uses_exact_budget_and_stays_in_domain(self) -> None:
        models = {target: LinearToyModel(target) for target in ("TN_out", "DEC")}
        context = dict.fromkeys(feature_names("TN_out"), 1.0)
        case = {"date": "2000-02-01", "context": context, "low": [0, 0], "high": [1, 1]}
        for method in SEARCH_METHODS:
            result = search_method(models, case, np.zeros(2), np.ones(2) * 3, method, 11, 64)
            self.assertEqual(result["n_eval"], 64)
            coordinates = np.asarray(result["front_u"])
            self.assertTrue(np.all((coordinates >= 0) & (coordinates <= 1)))
            self.assertEqual(len(nondominated(np.array(result["front_f"]))), result["front_points"])

    def test_invalid_budgets_fail_before_a_run(self) -> None:
        invalid: list[dict[str, Any]] = [
            {"evaluations": 63},
            {"rows": 0},
            {"optimizer_seeds": (11, 11)},
            {"scenarios": 0},
            {"seed": -1},
        ]
        for kwargs in invalid:
            with self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_same_domain_reference_is_projected_and_distinct(self) -> None:
        models = {"TN_out": SignedToyModel(1), "DEC": SignedToyModel(-1)}
        context = dict.fromkeys(feature_names("TN_out"), 1.0)
        context.update(PPA=2.0, DO=2.0)
        case = {"date": "2000-01-04", "context": context, "low": [0, 0], "high": [1, 1]}
        records = [
            {
                "date": case["date"],
                "method": "NSGA2",
                "front_u": [[0, 0], [1, 0]],
                "front_f": [[0, 0], [1, -1]],
            }
        ]
        solution = same_domain_solutions(models, [case], records, np.ones(2))[0]
        self.assertEqual(solution["anchor_actions"], [1.0, 1.0])
        self.assertFalse(solution["historical_in_domain"])
        self.assertNotEqual(solution["delta_historical"], solution["delta_same_domain"])

    def test_existing_output_is_not_replaced(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path / "sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Settings(), path)
            self.assertEqual((path / "sentinel.txt").read_text(encoding="utf-8"), "keep")

    def test_complete_run_has_matching_evidence_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "fresh"
            result = run(
                Settings(rows=180, evaluations=64, scenarios=1, optimizer_seeds=(11,)), output
            )
            self.assertTrue(result["synthetic_only"])
            self.assertEqual(result["test_access_blocks"], 1)

            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "COMPLETE")
            for name, digest in manifest["files"].items():
                self.assertEqual(sha(output / name), digest)
            self.assertEqual(result["selection_lock_sha256"], sha(output / "selection_lock.json"))
            self.assertEqual(
                sum(row["runs"] for row in result["search_summary"]), len(SEARCH_METHODS)
            )


if __name__ == "__main__":
    unittest.main()
