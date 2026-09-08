"""Direct, context-conditioned bi-objective search of the frozen wastewater proxies.

All decision coordinates are model-ready three-day means, not control commands.
Original source snapshots and the historical RL evaluation are read-only inputs.
"""
from __future__ import annotations
import os
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
from pathlib import Path
import sys
import time
import json
import hashlib
import warnings
import random
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import qmc
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pymoo.core.problem import Problem
from pymoo.core.population import Population
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.moo.moead import ParallelMOEAD
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.indicators.hv import HV

PAPER = Path(__file__).resolve().parents[2]
ROOT = PAPER / "revision_20260905"
sys.path.insert(0, str(PAPER / "SourceCode" / "src"))
from taici.initial_dataset import TN_FEATURES, DEC_FEATURES, load_initial_dataset

METHODS = ("NSGA2", "SPEA2", "MOEAD")
CONTEXT = ("Q", "COD", "TN_in", "NH3N", "T", "MLSS")
ACTIONS = ("PPA", "DO")

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

def nondominated_indices(F: np.ndarray) -> np.ndarray:
    """Exact two-objective minimization front, retaining one copy of each point."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return np.array([], dtype=int)
    if F.ndim != 2 or F.shape[1] != 2 or not np.isfinite(F).all():
        raise ValueError("Expected a finite two-objective matrix")
    order = np.lexsort((F[:, 1], F[:, 0]))
    keep, best = [], np.inf
    for i in order:
        if F[i, 1] < best:
            keep.append(i)
            best = F[i, 1]
    return np.asarray(keep, dtype=int)

def hypervolume(F: np.ndarray, ref: np.ndarray | tuple = (1.1, 1.1)) -> float:
    return float(HV(ref_point=np.asarray(ref, float))(np.asarray(F)[nondominated_indices(F)]))

def igd_plus(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference, candidate = np.asarray(reference), np.asarray(candidate)
    return float(np.sqrt(np.sum(np.maximum(candidate[None, :, :] - reference[:, None, :], 0.0)**2, axis=2)).min(axis=1).mean())

def configurations() -> list[dict]:
    result = []
    for pop in (32, 64):
        for profile, eta_c, eta_m, fraction, mating in (
            ("explore", 5, 5, .40, .7),
            ("balanced", 15, 20, .20, .9),
            ("local", 30, 40, .10, 1.0),
        ):
            result.append(dict(id=f"p{pop}_{profile}", pop_size=pop, crossover_probability=.9,
                               crossover_eta=eta_c, mutation_eta=eta_m, mutation_probability_per_variable=.5,
                               moead_neighbors=max(3, round(pop*fraction)), moead_neighbor_mating=mating))
    return result

def load_proxy():
    import joblib
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    threadpool_limits(1)
    warnings.filterwarnings("ignore", message="X does not have valid feature names")
    bundle = joblib.load(PAPER / "results/final_project/final_proxy_bundle.joblib")
    for model in [bundle.dec_model, *bundle.tn_model.base_models.values()]:
        if hasattr(model, "get_params"):
            updates = {k: 1 for k in model.get_params() if k.endswith("n_jobs") or k.endswith("thread_count")}
            if updates:
                model.set_params(**updates)
    return bundle

def predict(bundle, X: pd.DataFrame, chunk: int = 2048) -> np.ndarray:
    parts = []
    for start in range(0, len(X), chunk):
        block = X.iloc[start:start+chunk]
        parts.append(np.column_stack([bundle.predict_tn(block.loc[:, TN_FEATURES]), bundle.predict_dec(block.loc[:, DEC_FEATURES])]))
    result = np.vstack(parts)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite proxy output")
    return result

def prepare_contexts(frame: pd.DataFrame, per_month: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit support, imputation and bounds solely on 2023-2024 predictor records."""
    frame = frame.copy()
    frame["Date"] = pd.to_datetime(frame["Date"])
    train_mask = frame.Date < pd.Timestamp("2025-01-01")
    medians = frame.loc[train_mask, TN_FEATURES].median()
    frame.loc[:, TN_FEATURES] = frame.loc[:, TN_FEATURES].fillna(medians)
    train = frame.loc[train_mask]
    scaler = StandardScaler().fit(train.loc[:, CONTEXT])
    train_values = scaler.transform(train.loc[:, CONTEXT])
    nn = NearestNeighbors(n_neighbors=21).fit(train_values)
    distances, _ = nn.kneighbors(train_values)
    threshold = float(np.quantile(distances[:, 1:11].mean(axis=1), .95))
    global_low, global_high = np.quantile(train.loc[:, ACTIONS], [.05, .95], axis=0)
    target = frame.loc[~train_mask].copy()
    distances, indices = nn.kneighbors(scaler.transform(target.loc[:, CONTEXT]))
    target["context_distance"] = distances[:, :10].mean(axis=1)
    target["supported"] = target.context_distance <= threshold
    for j, action in enumerate(ACTIONS):
        local = train[action].to_numpy()[indices[:, :20]]
        target[f"low_{action}"] = np.maximum(np.quantile(local, .1, axis=1), global_low[j])
        target[f"high_{action}"] = np.minimum(np.quantile(local, .9, axis=1), global_high[j])
        target["supported"] &= target[f"high_{action}"] > target[f"low_{action}"] + 1e-12
    target["period"] = np.where(target.Date < pd.Timestamp("2025-07-01"), "tune", "evaluate")
    target["selected"] = False
    per_month = per_month or {"tune": 2, "evaluate": 4}
    for (_, period), group in target.loc[target.supported].groupby([target.loc[target.supported].Date.dt.month, "period"]):
        # Equally spaced quantiles of supported calendar dates; no outcomes enter selection.
        n = min(per_month[period], len(group))
        positions = np.unique(np.rint(np.linspace(0, len(group)-1, n)).astype(int))
        target.loc[group.index[positions], "selected"] = True
    metadata = dict(train_rows=int(train_mask.sum()), context_features=list(CONTEXT), threshold=threshold,
                    scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
                    global_low=global_low.tolist(), global_high=global_high.tolist(),
                    imputation=medians.to_dict(), support_neighbors=10, conditional_neighbors=20,
                    conditional_quantiles=[.1,.9], global_quantiles=[.05,.95],
                    selection="equally spaced supported dates within each month; no outcome criterion")
    return target.reset_index(drop=True), metadata

def feature_matrix(case: dict, U: np.ndarray) -> pd.DataFrame:
    U = np.asarray(U, float)
    if U.ndim != 2 or U.shape[1] != 2 or not np.isfinite(U).all() or np.any(U < -1e-10) or np.any(U > 1+1e-10):
        raise ValueError("Search coordinates must be finite in [0, 1]^2")
    X = pd.DataFrame(np.tile([case[f] for f in TN_FEATURES], (len(U), 1)), columns=TN_FEATURES)
    for j, action in enumerate(ACTIONS):
        X[action] = case[f"low_{action}"] + U[:, j]*(case[f"high_{action}"]-case[f"low_{action}"])
    return X

class BiObjectiveProblem(Problem):
    def __init__(self):
        super().__init__(n_var=2, n_obj=2, xl=np.zeros(2), xu=np.ones(2))
    def _evaluate(self, X, out, *args, **kwargs):
        raise RuntimeError("Evaluate through the shared exact proxy batch scheduler")

def make_algorithm(method: str, cfg: dict):
    kwargs = dict(sampling=LHS(), crossover=SBX(prob=cfg["crossover_probability"], eta=cfg["crossover_eta"]),
                  mutation=PM(prob=1.0, prob_var=cfg["mutation_probability_per_variable"], eta=cfg["mutation_eta"]))
    if method == "NSGA2":
        return NSGA2(pop_size=cfg["pop_size"], **kwargs)
    if method == "SPEA2":
        return SPEA2(pop_size=cfg["pop_size"], **kwargs)
    if method == "MOEAD":
        return ParallelMOEAD(get_reference_directions("uniform", 2, n_partitions=cfg["pop_size"]-1),
                             n_neighbors=cfg["moead_neighbors"], prob_neighbor_mating=cfg["moead_neighbor_mating"],
                             decomposition=Tchebicheff(), n_offsprings=cfg["pop_size"], **kwargs)
    raise ValueError(method)

@dataclass
class Job:
    case: dict
    method: str
    cfg: dict
    seed: int
    budget: int
    algorithm: Any = None
    rng: Any = None
    python_rng: Any = None
    all_u: list = field(default_factory=list)
    all_f: list = field(default_factory=list)
    history: list = field(default_factory=list)
    n: int = 0
    seconds: float = 0.0
    def __post_init__(self):
        # pymoo constructors contain stateful default survival/operator objects.
        # Match minimize(copy_algorithm=True) and isolate each independent run.
        self.algorithm = deepcopy(make_algorithm(self.method, self.cfg))
        self.algorithm.setup(BiObjectiveProblem(), termination=("n_eval", self.budget), seed=self.seed, verbose=False)
        self.rng = np.random.get_state()
        self.python_rng = random.getstate()
    def ask(self):
        np.random.set_state(self.rng)
        random.setstate(self.python_rng)
        t = time.perf_counter()
        infill = self.algorithm.ask()
        self.seconds += time.perf_counter()-t
        self.rng = np.random.get_state()
        self.python_rng = random.getstate()
        if infill is None or self.n + len(infill) > self.budget:
            raise RuntimeError("Algorithm failed to honor exact evaluation budget")
        return infill
    def tell(self, infill, raw, low, span):
        np.random.set_state(self.rng)
        random.setstate(self.python_rng)
        scaled = (raw-low)/span
        infill.set("F", scaled)
        for individual in infill:
            individual.evaluated.update(["F", "G", "H"])
        self.algorithm.evaluator.n_eval += len(infill)
        t = time.perf_counter()
        self.algorithm.tell(infills=infill)
        self.seconds += time.perf_counter()-t
        self.rng = np.random.get_state()
        self.python_rng = random.getstate()
        self.all_u.append(infill.get("X").copy())
        self.all_f.append(raw.copy())
        self.n += len(infill)
        if self.n % 256 == 0 or self.n == self.budget:
            F = (np.vstack(self.all_f)-low)/span
            self.history.append(dict(n_eval=self.n, hv=hypervolume(F), archive_size=len(nondominated_indices(F))))
    def result(self, low, span):
        U, F = np.vstack(self.all_u), np.vstack(self.all_f)
        keep = nondominated_indices(F)
        return dict(date=str(self.case["Date"])[:10], method=self.method, config=self.cfg["id"], seed=self.seed,
                    n_eval=self.n, seconds=self.seconds, hv=hypervolume((F-low)/span),
                    front_u=U[keep].tolist(), front_f=F[keep].tolist(), history=self.history,
                    outside_reference_fraction=float(np.any((F-low)/span >=1.1,axis=1).mean()))

def run_batch(bundle, cases: list[dict], method: str, cfg: dict, seed: int, budget: int, low, span) -> list[dict]:
    """Batch independent contexts without changing any algorithm's random stream."""
    jobs = [Job(case, method, cfg, seed, budget) for case in cases]
    while any(job.n < budget for job in jobs):
        active = [job for job in jobs if job.n < budget]
        infills = [job.ask() for job in active]
        matrices = [feature_matrix(job.case, pop.get("X")) for job, pop in zip(active, infills)]
        t = time.perf_counter()
        F = predict(bundle, pd.concat(matrices, ignore_index=True))
        elapsed = time.perf_counter()-t
        offset = 0
        for job, pop in zip(active, infills):
            job.seconds += elapsed*len(pop)/len(F)
            job.tell(pop, F[offset:offset+len(pop)], low, span)
            offset += len(pop)
    return [job.result(low, span) for job in jobs]
