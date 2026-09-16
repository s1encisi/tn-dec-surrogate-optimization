"""Synthetic prediction, model interpretation, support-constrained Pareto comparison."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, StackingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.core.problem import Problem
from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.util.ref_dirs import get_reference_directions

CONTINUOUS = ["Q", "COD", "TN_in", "NH3N", "T", "PPA", "DO", "MLSS"]
FEATURES = ["doy_sin", "doy_cos", "trend", *CONTINUOUS]
TARGETS = ["TN_out", "DEC"]
ACTIONS = ["PPA", "DO"]
CONTEXT = ["Q", "COD", "TN_in", "NH3N", "T", "MLSS"]


@dataclass(frozen=True)
class Settings:
    seed: int = 17
    rows: int = 480
    evaluations: int = 256
    optimizer_seeds: tuple[int, ...] = (11, 23, 37)
    scenarios: int = 3

    def __post_init__(self):
        if type(self.seed) is not int or not 0 <= self.seed <= 10_000_000:
            raise ValueError("Invalid seed.")
        if type(self.rows) is not int or not 180 <= self.rows <= 2000:
            raise ValueError("rows must be between 180 and 2000.")
        if type(self.evaluations) is not int or self.evaluations < 64 or self.evaluations > 4096 or self.evaluations % 32:
            raise ValueError("evaluations must be a multiple of 32 in [64, 4096].")
        if not 1 <= self.scenarios <= 6:
            raise ValueError("scenarios must be between 1 and 6.")
        if not self.optimizer_seeds or len(set(self.optimizer_seeds)) != len(self.optimizer_seeds):
            raise ValueError("Optimizer seeds must be nonempty and unique.")


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synthetic_data(rows: int, seed: int) -> pd.DataFrame:
    """All ranges and equations below are invented for this demonstration."""
    rng = np.random.default_rng(seed)
    n = rows+3
    day = np.arange(n)
    raw = pd.DataFrame({
        "Q": rng.uniform(15000, 25000, n),
        "COD": rng.uniform(100, 300, n),
        "TN_in": rng.uniform(15, 40, n),
        "NH3N": rng.uniform(10, 30, n),
        "T": 20+6*np.sin(2*np.pi*day/365.2425)+rng.normal(0, 1, n),
        "PPA": rng.uniform(300, 900, n),
        "DO": rng.uniform(1.3, 2.9, n),
        "MLSS": rng.uniform(3000, 5000, n),
    })
    frame = raw.shift(1).rolling(3, min_periods=3).mean().iloc[3:].copy()
    frame.insert(0, "Date", pd.date_range("2000-01-04", periods=rows))
    doy = frame.Date.dt.dayofyear.to_numpy()
    frame["doy_sin"] = np.sin(2*np.pi*doy/365.2425)
    frame["doy_cos"] = np.cos(2*np.pi*doy/365.2425)
    frame["trend"] = np.arange(rows)
    frame["TN_out"] = (7+.12*(frame.TN_in-25)-.12*(frame["T"]-20)
                       +.0002*(frame.MLSS-4000)+.0012*(frame.PPA-600)
                       -.6*(frame.DO-2)+.000002*(frame.PPA-600)**2
                       +rng.normal(0, .10, rows))
    frame["DEC"] = (8500+.08*(frame.Q-20000)+.45*(frame.MLSS-4000)
                    +500*(frame.DO-2)+.006*(frame.PPA-550)**2
                    +rng.normal(0, 80, rows))
    # Deliberate missing feature values exercise fold-fitted imputation.
    frame.loc[frame.index[::79], "DO"] = np.nan
    return frame.reset_index(drop=True)


def splits(rows: int, seed: int) -> dict[str, np.ndarray]:
    development, test = train_test_split(np.arange(rows), test_size=.2, random_state=seed)
    train, validation = train_test_split(development, test_size=.25, random_state=seed+1)
    result = {"train": np.sort(train), "validation": np.sort(validation), "test": np.sort(test)}
    if any(set(result[a]) & set(result[b]) for a, b in
           [("train", "validation"), ("train", "test"), ("validation", "test")]):
        raise ValueError("Data splits overlap.")
    return result


def registry(seed: int):
    def imputer():
        return SimpleImputer(strategy="median", add_indicator=True)
    ridge = make_pipeline(imputer(), PolynomialFeatures(2, include_bias=False),
                          StandardScaler(), Ridge(alpha=1.0))
    extra = make_pipeline(imputer(), ExtraTreesRegressor(
        n_estimators=96, min_samples_leaf=2, random_state=seed, n_jobs=1))
    forest = make_pipeline(imputer(), RandomForestRegressor(
        n_estimators=48, min_samples_leaf=2, random_state=seed, n_jobs=1))
    stack = StackingRegressor(
        estimators=[("ridge", ridge), ("extra", extra), ("forest", forest)],
        final_estimator=make_pipeline(StandardScaler(), HuberRegressor(max_iter=1000)),
        cv=KFold(3, shuffle=True, random_state=seed), n_jobs=1)
    return {"TrainingMean": make_pipeline(imputer(), DummyRegressor()),
            "RidgePolynomial": ridge, "ExtraTrees": extra, "HuberStack": stack}


def feature_names(target: str):
    return [name for name in FEATURES if target != "DEC" or name != "TN_in"]


def metric(y, predicted):
    return {"R2": float(r2_score(y, predicted)),
            "RMSE": float(root_mean_squared_error(y, predicted)),
            "MAE": float(mean_absolute_error(y, predicted))}


def fit_select(frame: pd.DataFrame, split: dict, seed: int):
    """Selection reads training/validation only. Test targets cannot influence this function."""
    train, validation = split["train"], split["validation"]
    trials, selected, models, interpretation_models = [], {}, {}, {}
    development = np.concatenate([train, validation])
    for target in TARGETS:
        names = feature_names(target)
        fitted = {}
        for name, estimator in registry(seed).items():
            start = time.perf_counter()
            fitted[name] = clone(estimator).fit(frame.loc[train, names], frame.loc[train, target])
            result = metric(frame.loc[validation, target],
                            fitted[name].predict(frame.loc[validation, names]))
            trials.append({"target": target, "model": name, **result,
                           "fit_validation_seconds": time.perf_counter()-start})
        eligible = [r for r in trials if r["target"] == target and r["model"] != "TrainingMean"]
        chosen = min(eligible, key=lambda r: (r["RMSE"], r["model"]))["model"]
        selected[target] = chosen
        interpretation_models[target] = fitted[chosen]
        models[target] = clone(fitted[chosen]).fit(
            frame.loc[development, names], frame.loc[development, target])
    return selected, models, interpretation_models, trials


def objective(models: dict, frame: pd.DataFrame):
    return np.column_stack([models[t].predict(frame.loc[:, feature_names(t)]) for t in TARGETS])


def nondominated(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-objective outputs.")
    best, keep = float("inf"), []
    for i in np.lexsort((values[:, 1], values[:, 0])):
        if values[i, 1] < best:
            keep.append(i)
            best = values[i, 1]
    return np.asarray(keep, int)


def build_cases(frame: pd.DataFrame, development: np.ndarray, count: int):
    train = frame.loc[development].copy()
    fill = train[FEATURES].median()
    train.loc[:, FEATURES] = train[FEATURES].fillna(fill)
    scaler = StandardScaler().fit(train[CONTEXT])
    values = scaler.transform(train[CONTEXT])
    nn = NearestNeighbors(n_neighbors=21).fit(values)
    indices = np.unique(np.rint(np.linspace(0, len(train)-1, count)).astype(int))
    ordered = train.sort_values("T")
    global_low, global_high = train[ACTIONS].quantile([.05, .95]).to_numpy()
    cases = []
    for _, row in ordered.iloc[indices].iterrows():
        _, nearby = nn.kneighbors(scaler.transform(row[CONTEXT].to_frame().T), n_neighbors=21)
        # Exclude the exact self-match to define a historical-neighbour rectangle.
        local = train.iloc[nearby[0][1:]][ACTIONS]
        low = np.maximum(local.quantile(.1).to_numpy(), global_low)
        high = np.minimum(local.quantile(.9).to_numpy(), global_high)
        if np.any(high <= low):
            raise ValueError("Empty conditional support interval.")
        cases.append({"date": row.Date.strftime("%Y-%m-%d"),
                      "context": {key: float(row[key]) for key in FEATURES},
                      "low": low.tolist(), "high": high.tolist()})
    return cases, fill


class ProxyProblem(Problem):
    def __init__(self, models, case, offset, scale):
        super().__init__(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2))
        self.models, self.case, self.offset, self.scale = models, case, offset, scale
        self.all_u, self.all_f = [], []

    def raw(self, unit):
        x = pd.DataFrame([self.case["context"]]*len(unit))
        low, high = np.asarray(self.case["low"]), np.asarray(self.case["high"])
        x.loc[:, ACTIONS] = low+np.asarray(unit)*(high-low)
        return objective(self.models, x)

    def _evaluate(self, unit, out, *args, **kwargs):
        raw = self.raw(unit)
        self.all_u.append(np.asarray(unit).copy())
        self.all_f.append(raw.copy())
        out["F"] = (raw-self.offset)/self.scale


def search_method(models, case, offset, scale, method: str, seed: int, budget: int):
    problem = ProxyProblem(models, case, offset, scale)
    start = time.perf_counter()
    if method == "UniformRandom":
        unit = np.random.default_rng(seed).random((budget, 2))
        raw = problem.raw(unit)
    else:
        variation = {"sampling": LHS(), "crossover": SBX(prob=.9, eta=15),
                     "mutation": PM(prob=1, prob_var=.5, eta=20)}
        if method == "NSGA2":
            algorithm = NSGA2(pop_size=32, **variation)
        elif method == "SPEA2":
            algorithm = SPEA2(pop_size=32, **variation)
        elif method == "MOEAD":
            algorithm = ParallelMOEAD(
                get_reference_directions("uniform", 2, n_partitions=31),
                n_neighbors=13, prob_neighbor_mating=.7, n_offsprings=32, **variation)
        else:
            raise ValueError("Unknown search method.")
        minimize(problem, algorithm, ("n_eval", budget), seed=seed, verbose=False)
        unit, raw = np.vstack(problem.all_u), np.vstack(problem.all_f)
    elapsed = time.perf_counter()-start
    if len(raw) != budget:
        raise ValueError("Search method did not use the exact registered budget.")
    keep = nondominated(raw)
    front, coordinates = raw[keep], unit[keep]
    if np.any(coordinates < -1e-12) or np.any(coordinates > 1+1e-12):
        raise ValueError("Candidate escaped the registered support rectangle.")
    normalized = (front-offset)/scale
    hv = float(HV(ref_point=np.array([1.1, 1.1]))(normalized))
    return {"method": method, "seed": seed, "date": case["date"], "n_eval": len(raw),
            "HV": hv, "seconds": elapsed, "front_points": len(front),
            "front_u": coordinates.tolist(), "front_f": front.tolist()}




def same_domain_solutions(models, cases, records, scale):
    """Compare an NSGA-II compromise with both historical and projected references."""
    solutions = []
    for case in cases:
        selected = [r for r in records if r["date"] == case["date"] and r["method"] == "NSGA2"]
        unit = np.vstack([r["front_u"] for r in selected])
        values = np.vstack([r["front_f"] for r in selected])
        keep = nondominated(values)
        unit, values = unit[keep], values[keep]
        chosen = int(np.argmin(np.linalg.norm((values-values.min(axis=0))/scale, axis=1)))
        low, high = np.asarray(case["low"]), np.asarray(case["high"])
        historical = pd.DataFrame([case["context"]])
        anchor = historical.copy()
        original_actions = historical[ACTIONS].iloc[0].to_numpy()
        anchor.loc[:, ACTIONS] = np.clip(original_actions, low, high)[None, :]
        baseline, same_domain = objective(models, historical)[0], objective(models, anchor)[0]
        solutions.append({
            "date": case["date"], "method": "NSGA2",
            "compromise": values[chosen].tolist(),
            "compromise_actions": (low+unit[chosen]*(high-low)).tolist(),
            "anchor_actions": anchor[ACTIONS].iloc[0].tolist(),
            "same_domain_reference": same_domain.tolist(),
            "historical_reference": baseline.tolist(),
            "historical_in_domain": bool(np.all(original_actions >= low) and np.all(original_actions <= high)),
            "delta_historical": (values[chosen]-baseline).tolist(),
            "delta_same_domain": (values[chosen]-same_domain).tolist(),
        })
    return solutions

def run(settings: Settings, output: Path):
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    phase = "generate"
    try:
        frame = synthetic_data(settings.rows, settings.seed)
        split = splits(len(frame), settings.seed)
        protocol = {"schema": "synthetic_demo_v1", "synthetic_only": True,
                    "settings": asdict(settings), "targets": TARGETS,
                    "continuous_input_window": "mean(x[t-1], x[t-2], x[t-3])",
                    "selection": "validation RMSE; TrainingMean excluded from selection",
                    "optimizer_parameters": "fixed demonstration parameters; no tuning",
                    "explanation_method": "NSGA2 predeclared for same-domain examples, not selected from test results",
                    "split": {k: v.tolist() for k, v in split.items()},
                    "packages": {p: version(p) for p in ("numpy", "pandas", "scikit-learn", "pymoo")},
                    "data_sha256": hashlib.sha256(
                        frame.to_csv(index=False, float_format="%.17g").encode()).hexdigest()}
        (output/"protocol.json").write_bytes(canonical(protocol))
        phase = "fit_select"
        selected, models, interpretation, trials = fit_select(frame, split, settings.seed)
        lock = {"selected": selected, "test_used_for_selection": False,
                "protocol_sha256": sha(output/"protocol.json")}
        (output/"selection_lock.json").write_bytes(canonical(lock))
        lock_hash = sha(output/"selection_lock.json")
        phase = "sealed_test"
        test_rows = split["test"]
        test = {target: metric(frame.loc[test_rows, target],
                              models[target].predict(frame.loc[test_rows, feature_names(target)]))
                for target in TARGETS}
        phase = "interpretation"
        importance = {}
        for target in TARGETS:
            names = feature_names(target)
            pfi = permutation_importance(
                interpretation[target], frame.loc[split["validation"], names],
                frame.loc[split["validation"], target], n_repeats=3,
                random_state=settings.seed, scoring="neg_root_mean_squared_error", n_jobs=1)
            importance[target] = dict(zip(names, pfi.importances_mean.tolist()))
        development = np.concatenate([split["train"], split["validation"]])
        cases, fill = build_cases(frame, development, settings.scenarios)
        fit_frame = frame.loc[development, FEATURES].fillna(fill)
        low, high = np.quantile(objective(models, fit_frame), [.05, .95], axis=0)
        span = np.maximum(high-low, 1e-8)
        phase = "pareto_search"
        records = [search_method(models, case, low, span, method, seed, settings.evaluations)
                   for case in cases for method in ("NSGA2", "SPEA2", "MOEAD", "UniformRandom")
                   for seed in settings.optimizer_seeds]
        solutions = same_domain_solutions(models, cases, records, span)
        summaries = []
        for method in ("NSGA2", "SPEA2", "MOEAD", "UniformRandom"):
            part = [r for r in records if r["method"] == method]
            seed_hv = [np.mean([r["HV"] for r in part if r["seed"] == seed])
                       for seed in settings.optimizer_seeds]
            timing = np.asarray([r["seconds"] for r in part])
            summaries.append({"method": method, "HV_seed_mean": float(np.mean(seed_hv)),
                              "HV_seed_sd": float(np.std(seed_hv, ddof=1)) if len(seed_hv)>1 else None,
                              "seconds_p50": float(np.quantile(timing, .5)),
                              "seconds_p95": float(np.quantile(timing, .95)),
                              "runs": len(part), "total_evaluations": sum(r["n_eval"] for r in part)})
        assert sha(output/"selection_lock.json") == lock_hash
        result = {"status": "COMPLETE", "synthetic_only": True, "selected": selected,
                  "test": test, "validation_trials": trials, "importance": importance,
                  "search_summary": summaries, "cases": cases, "solutions": solutions,
                  "runtime_seconds": time.perf_counter()-started,
                  "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "test_access_blocks": 1, "selection_lock_sha256": lock_hash,
                  "scientific_scope": "synthetic workflow demonstration; no factory or causal claim"}
        (output/"summary.json").write_bytes(canonical(result))
        (output/"pareto.json").write_bytes(canonical(records))
        from .report import build_report
        build_report(result, records, output)
        manifest = {p.name: sha(p) for p in output.iterdir() if p.is_file()}
        (output/"manifest.json").write_bytes(canonical({"status": "COMPLETE", "files": manifest}))
        return result
    except Exception:
        (output/"failure.json").write_bytes(canonical({"status": "FAILED", "phase": phase}))
        raise
