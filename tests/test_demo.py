from pathlib import Path
import json
import tempfile
import unittest

import numpy as np
import pandas as pd

from tn_dec_demo.workflow import (
    Settings, synthetic_data, splits, registry, fit_select, feature_names,
    nondominated, search_method, same_domain_solutions, run, sha,
)


class DemoTests(unittest.TestCase):
    def test_synthetic_fixture_reproducibility_and_missingness(self):
        first = synthetic_data(180, 17)
        pd.testing.assert_frame_equal(first, synthetic_data(180, 17))
        self.assertFalse(first.equals(synthetic_data(180, 18)))
        self.assertTrue(first["DO"].isna().any())
        self.assertFalse(first[["TN_out", "DEC"]].isna().any().any())
        self.assertEqual(str(first.Date.iloc[0].date()), "2000-01-04")

    def test_split_is_a_disjoint_complete_partition(self):
        parts = splits(480, 17)
        sets = [set(v) for v in parts.values()]
        self.assertEqual([len(s) for s in sets], [288, 96, 96])
        self.assertEqual(set.union(*sets), set(range(480)))
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])

    def test_test_data_cannot_change_selection_or_fitted_models(self):
        frame = synthetic_data(180, 17)
        parts = splits(len(frame), 17)
        first, models, _, trials = fit_select(frame, parts, 17)
        changed = frame.copy()
        changed.loc[parts["test"], feature_names("TN_out")] = 1e7
        changed.loc[parts["test"], ["TN_out", "DEC"]] = -1e8
        second, models2, _, _ = fit_select(changed, parts, 17)
        self.assertEqual(first, second)
        for target in first:
            x = frame.loc[parts["validation"], feature_names(target)]
            np.testing.assert_allclose(models[target].predict(x), models2[target].predict(x),
                                       atol=1e-9, rtol=1e-9)
            candidates = [r for r in trials if r["target"] == target and r["model"] != "TrainingMean"]
            self.assertEqual(first[target], min(candidates, key=lambda r: (r["RMSE"], r["model"]))["model"])

    def test_preprocessing_statistics_are_fitted_on_training_rows(self):
        frame = synthetic_data(180, 29)
        parts = splits(len(frame), 29)
        columns = feature_names("TN_out")
        estimator = registry(29)["RidgePolynomial"].fit(
            frame.loc[parts["train"], columns], frame.loc[parts["train"], "TN_out"])
        actual = estimator.named_steps["simpleimputer"].statistics_
        expected = frame.loc[parts["train"], columns].median().to_numpy()
        np.testing.assert_allclose(actual, expected)

    def test_pareto_archive_handles_duplicates_and_dominance(self):
        f = np.array([[0, 1], [1, 0], [.5, .5], [.75, .75], [.5, .5]])
        self.assertEqual(set(nondominated(f)), {0, 1, 2})
        with self.assertRaises(ValueError):
            nondominated(np.array([[float("nan"), 1]]))

    def test_each_search_method_uses_exact_budget_and_stays_in_domain(self):
        class Toy:
            def __init__(self, target):
                self.target = target
            def predict(self, x):
                if self.target == "TN_out":
                    return x.PPA.to_numpy()+.1*x.DO.to_numpy()
                return 2-x.PPA.to_numpy()+.1*x.DO.to_numpy()
        models = {t: Toy(t) for t in ["TN_out", "DEC"]}
        context = {k: 1.0 for k in feature_names("TN_out")}
        case = {"date": "2000-02-01", "context": context, "low": [0, 0], "high": [1, 1]}
        for method in ["NSGA2", "SPEA2", "MOEAD", "UniformRandom"]:
            result = search_method(models, case, np.zeros(2), np.ones(2)*3, method, 11, 64)
            self.assertEqual(result["n_eval"], 64)
            coordinates = np.asarray(result["front_u"])
            self.assertTrue(np.all((coordinates >= 0) & (coordinates <= 1)))
            self.assertEqual(len(nondominated(np.array(result["front_f"]))), result["front_points"])

    def test_invalid_budgets_fail_before_a_run(self):
        for kwargs in [{"evaluations": 63}, {"rows": 0}, {"optimizer_seeds": (11, 11)},
                       {"scenarios": 0}, {"seed": -1}]:
            with self.assertRaises(ValueError):
                Settings(**kwargs)

    def test_same_domain_reference_is_projected_and_distinct(self):
        class Toy:
            def __init__(self, sign): self.sign = sign
            def predict(self, x): return self.sign*x.PPA.to_numpy()+x.DO.to_numpy()
        models = {"TN_out": Toy(1), "DEC": Toy(-1)}
        context = {k: 1.0 for k in feature_names("TN_out")}
        context.update(PPA=2.0, DO=2.0)
        case = {"date": "2000-01-04", "context": context, "low": [0, 0], "high": [1, 1]}
        records = [{"date": case["date"], "method": "NSGA2",
                    "front_u": [[0, 0], [1, 0]], "front_f": [[0, 0], [1, -1]]}]
        solution = same_domain_solutions(models, [case], records, np.ones(2))[0]
        self.assertEqual(solution["anchor_actions"], [1.0, 1.0])
        self.assertFalse(solution["historical_in_domain"])
        self.assertNotEqual(solution["delta_historical"], solution["delta_same_domain"])

    def test_existing_output_is_not_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            (path/"sentinel.txt").write_text("keep", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(Settings(), path)
            self.assertEqual((path/"sentinel.txt").read_text(), "keep")

    def test_complete_run_has_matching_evidence_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/"fresh"
            result = run(Settings(rows=180, evaluations=64, scenarios=1,
                                  optimizer_seeds=(11,)), output)
            self.assertTrue(result["synthetic_only"])
            self.assertEqual(result["test_access_blocks"], 1)
            manifest = json.loads((output/"manifest.json").read_text())
            self.assertEqual(manifest["status"], "COMPLETE")
            for name, digest in manifest["files"].items():
                self.assertEqual(sha(output/name), digest)
            self.assertEqual(result["selection_lock_sha256"], sha(output/"selection_lock.json"))
            self.assertEqual(sum(r["runs"] for r in result["search_summary"]), 4)


if __name__ == "__main__":
    unittest.main()
