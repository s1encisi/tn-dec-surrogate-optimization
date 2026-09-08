from __future__ import annotations
import argparse
import json
import time
import platform
import importlib.metadata
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from context_moo import *

def protocol():
    return dict(version="context_moo_v1", decision_variables=list(ACTIONS),
                decision_semantics="model-ready three-day-mean sensitivity coordinates",
                independent_predictive_test=False, plant_control=False,
                tuning_start="2025-01-04", tuning_end="2025-06-30",
                evaluation_start="2025-07-01", evaluation_end="2025-12-28",
                constraints_fit_end="2024-12-31", methods=list(METHODS), configurations=configurations(),
                tuning_seeds=[11,23,37], evaluation_seeds=[101,211,307,401,503],
                tuning_budget=1024, development_budgets=[2048,4096],
                minimum_final_budget=2048, maximum_budget=4096,
                plateau_rule="all methods median relative archive HV gain from half to full budget <=0.001 on tuning cases; otherwise extend all to 4096",
                selection="maximize mean per-case archive HV then minimize across-case-seed HV standard deviation; lexical config id breaks exact ties",
                front="external nondominated archive of all evaluated candidates for every method",
                evaluation_reference=[1.1,1.1], reference_sensitivity=[1.05,1.2],
                grid_reference_size=65, fixed_do_grid_size=257, local_grid_refinement_size=129,
                baseline_methods=["UniformRandom","LatinHypercube"],
                moead_variant="pymoo ParallelMOEAD synchronous offspring generation and Tchebycheff decomposition",
                inference="all frozen fitted components; concatenated batches of independent scenarios; one CPU thread",
                date_selection="2 tuning and 4 evaluation supported dates per calendar month, equally spaced; no outcomes used")

def prepare():
    out = ROOT / "results"
    if (out / "protocol.json").exists():
        raise RuntimeError("Protocol already exists; inspect or resume existing work")
    bundle = load_proxy()
    dataset = load_initial_dataset(PAPER / "InitialData/initial_dataset.csv")
    frame = dataset.frame.copy()
    cases, meta = prepare_contexts(frame)
    fit_frame = frame.loc[pd.to_datetime(frame.Date) < "2025-01-01", TN_FEATURES].copy()
    fit_frame = fit_frame.fillna(fit_frame.median())
    predictions = predict(bundle, fit_frame)
    low, high = np.quantile(predictions, [.05,.95], axis=0)
    if np.any(high <= low):
        raise ValueError("Degenerate objective scales")
    out.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out / "scenario_registry.csv", index=False)
    meta.update(objective_low=low.tolist(), objective_high=high.tolist())
    write_json(out / "context_contract.json", meta)
    value = protocol()
    value["inputs"] = {str(p.relative_to(PAPER)):sha(p) for p in [
        PAPER/"InitialData/initial_dataset.csv", PAPER/"results/final_project/final_proxy_bundle.joblib",
        Path(__file__), Path(__file__).with_name("context_moo.py")]}
    value["original_documents"] = {str(p.relative_to(PAPER)):sha(p) for p in [PAPER/"正文_中文初稿.docx", PAPER/"Supplementary_Information_中文初稿.docx"]}
    value["created_utc"] = datetime.now(timezone.utc).isoformat()
    value["environment"] = {x:importlib.metadata.version(x) for x in ["pymoo","numpy","pandas","scikit-learn","torch"]}
    value["environment"]["python"] = platform.python_version()
    write_json(out / "protocol.json", value)
    print(cases.groupby("period")[["supported","selected"]].sum().to_json(), flush=True)

def setup():
    out=ROOT/"results"
    p=json.loads((out/"protocol.json").read_text(encoding="utf-8"))
    for name, expected in p["inputs"].items():
        if sha(PAPER/name) != expected:
            raise RuntimeError(f"Frozen input changed: {name}")
    registry=pd.read_csv(out/"scenario_registry.csv")
    meta=json.loads((out/"context_contract.json").read_text(encoding="utf-8"))
    low=np.array(meta["objective_low"]); span=np.array(meta["objective_high"])-low
    return out,p,registry,low,span

def run_tuning():
    out,p,registry,low,span=setup()
    if (out/"selection_lock.json").exists():
        raise RuntimeError("Selection already frozen")
    cases=registry.loc[registry.selected & (registry.period=="tune")].to_dict("records")
    bundle=load_proxy()
    for method in METHODS:
        for cfg in p["configurations"]:
            for seed in p["tuning_seeds"]:
                path=out/"tuning"/f"{method}_{cfg['id']}_{seed}.json"
                if path.exists():
                    continue
                t=time.perf_counter()
                result=run_batch(bundle,cases,method,cfg,seed,p["tuning_budget"],low,span)
                write_json(path,result)
                print(json.dumps(dict(stage="tune",method=method,config=cfg["id"],seed=seed,cases=len(cases),seconds=round(time.perf_counter()-t,1),mean_hv=np.mean([r["hv"] for r in result]))),flush=True)
    rows=[r for path in sorted((out/"tuning").glob("*.json")) for r in json.loads(path.read_text())]
    tab=pd.DataFrame([{k:v for k,v in row.items() if k not in ["history","front_u","front_f"]} for row in rows])
    tab.to_csv(out/"tuning_trials.csv",index=False)
    summary=tab.groupby(["method","config"]).agg(mean_hv=("hv","mean"),std_hv=("hv","std"),seconds=("seconds","sum"),runs=("hv","size")).reset_index()
    summary.to_csv(out/"tuning_summary.csv",index=False)
    chosen={}
    for method in METHODS:
        record=summary[summary.method==method].sort_values(["mean_hv","std_hv","config"],ascending=[False,True,True]).iloc[0]
        chosen[method]=next(cfg for cfg in p["configurations"] if cfg["id"]==record["config"])
    # Common-budget convergence check only on the tuning scenarios.
    gates=[]
    selected_budget=4096
    for budget in p["development_budgets"]:
        all_gains=[]
        for method,cfg in chosen.items():
            method_gains=[]
            for seed in p["tuning_seeds"]:
                path=out/"development"/f"{method}_{cfg['id']}_{seed}_{budget}.json"
                if path.exists():
                    result=json.loads(path.read_text())
                else:
                    result=run_batch(bundle,cases,method,cfg,seed,budget,low,span)
                    write_json(path,result)
                for row in result:
                    half=next(h["hv"] for h in row["history"] if h["n_eval"]==budget//2)
                    gain=(row["hv"]-half)/max(abs(half),1e-12)
                    method_gains.append(gain)
                print(json.dumps(dict(stage="development",method=method,seed=seed,budget=budget)),flush=True)
            median=float(np.median(method_gains))
            gates.append(dict(method=method,budget=budget,median_relative_gain=median,max_relative_gain=float(max(method_gains)),fraction_below_threshold=float(np.mean(np.array(method_gains)<=.001))))
            all_gains.append(median)
        if max(all_gains)<=.001:
            selected_budget=budget
            break
    method_selection=summary[summary.apply(lambda r:r["config"]==chosen[r["method"]]["id"],axis=1)].sort_values(["mean_hv","std_hv","method"],ascending=[False,True,True])
    lock=dict(created_utc=datetime.now(timezone.utc).isoformat(),configurations=chosen,budget=selected_budget,
              primary_method=method_selection.iloc[0]["method"],selection_basis="tuning results only",convergence=gates,
              protocol_sha256=sha(out/"protocol.json"),tuning_summary_sha256=sha(out/"tuning_summary.csv"))
    write_json(out/"selection_lock.json",lock)
    print(json.dumps(lock),flush=True)

def evaluate():
    out,p,registry,low,span=setup()
    lock=json.loads((out/"selection_lock.json").read_text())
    if sha(out/"protocol.json")!=lock["protocol_sha256"] or sha(out/"tuning_summary.csv")!=lock["tuning_summary_sha256"]:
        raise RuntimeError("Selection lock mismatch")
    cases=registry.loc[registry.selected & (registry.period=="evaluate")].to_dict("records")
    bundle=load_proxy()
    budget=lock["budget"]
    for method,cfg in lock["configurations"].items():
        for seed in p["evaluation_seeds"]:
            path=out/"evaluation"/f"{method}_{seed}.json"
            if path.exists():
                continue
            t=time.perf_counter()
            result=run_batch(bundle,cases,method,cfg,seed,budget,low,span)
            write_json(path,result)
            print(json.dumps(dict(stage="evaluate",method=method,seed=seed,seconds=round(time.perf_counter()-t,1))),flush=True)
    for method in p["baseline_methods"]:
        for seed in p["evaluation_seeds"]:
            path=out/"evaluation"/f"{method}_{seed}.json"
            if path.exists():
                continue
            U=np.random.default_rng(seed).random((budget,2)) if method=="UniformRandom" else qmc.LatinHypercube(d=2,seed=seed).random(budget)
            t=time.perf_counter()
            F=predict(bundle,pd.concat([feature_matrix(c,U) for c in cases],ignore_index=True))
            elapsed=time.perf_counter()-t
            records=[]
            for i,case in enumerate(cases):
                raw=F[i*budget:(i+1)*budget]
                keep=nondominated_indices(raw)
                history=[dict(n_eval=n,hv=hypervolume((raw[:n]-low)/span),archive_size=len(nondominated_indices(raw[:n]))) for n in range(256,budget+1,256)]
                records.append(dict(date=str(case["Date"])[:10],method=method,config="untuned_space_filling",seed=seed,n_eval=budget,seconds=elapsed/len(cases),hv=hypervolume((raw-low)/span),front_u=U[keep].tolist(),front_f=raw[keep].tolist(),history=history,outside_reference_fraction=float(np.any((raw-low)/span>=1.1,axis=1).mean())))
            write_json(path,records)
            print(json.dumps(dict(stage="evaluate",method=method,seed=seed,seconds=round(elapsed,1))),flush=True)
    for case in cases:
        date=str(case["Date"])[:10]
        path=out/"reference"/f"{date}.json"
        if path.exists():
            continue
        grid=np.linspace(0,1,p["grid_reference_size"])
        U=np.stack(np.meshgrid(grid,grid),axis=-1).reshape(-1,2)
        raw=predict(bundle,feature_matrix(case,U))
        keep=nondominated_indices(raw)
        do=(case["DO"]-case["low_DO"])/(case["high_DO"]-case["low_DO"])
        fixed_supported=bool(0<=do<=1)
        fixedU=np.column_stack([np.linspace(0,1,p["fixed_do_grid_size"]),np.repeat(np.clip(do,0,1),p["fixed_do_grid_size"])])
        fixedF=predict(bundle,feature_matrix(case,fixedU))
        fk=nondominated_indices(fixedF)
        baseline=predict(bundle,pd.DataFrame([case],columns=TN_FEATURES))[0]
        value=dict(date=date,grid_size=len(U),grid_hv=hypervolume((raw-low)/span),front_u=U[keep].tolist(),front_f=raw[keep].tolist(),
                   historical_proxy=baseline.tolist(),historical_u=[(case[a]-case[f"low_{a}"])/(case[f"high_{a}"]-case[f"low_{a}"]) for a in ACTIONS],
                   fixed_do_is_historical_supported=fixed_supported,fixed_do_model_coordinate=float(case["low_DO"]+np.clip(do,0,1)*(case["high_DO"]-case["low_DO"])),
                   fixed_do_front_u=fixedU[fk].tolist(),fixed_do_front_f=fixedF[fk].tolist(),fixed_do_hv=hypervolume((fixedF-low)/span))
        write_json(path,value)
        np.savez_compressed(out/"reference"/f"{date}_surface.npz",U=U,F=raw)
        print(json.dumps(dict(stage="reference",date=date,grid=len(U))),flush=True)
    write_json(out/"run_status.json",dict(status="EXPERIMENTS_COMPLETE_PENDING_ANALYSIS",completed_utc=datetime.now(timezone.utc).isoformat(),selection_lock_sha256=sha(out/"selection_lock.json")))

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("phase",choices=["prepare","tune","evaluate"]);args=parser.parse_args()
    {"prepare":prepare,"tune":run_tuning,"evaluate":evaluate}[args.phase]()
