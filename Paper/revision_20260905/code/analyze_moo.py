"""Analyze locked evaluation outputs; no algorithm or parameter reselection."""
from context_moo import *
from datetime import datetime, timezone
OUT=ROOT/"results"
lock=json.loads((OUT/"selection_lock.json").read_text(encoding="utf-8"))
protocol=json.loads((OUT/"protocol.json").read_text(encoding="utf-8"))
contract=json.loads((OUT/"context_contract.json").read_text(encoding="utf-8"))
low=np.array(contract["objective_low"]);span=np.array(contract["objective_high"])-low
registry=pd.read_csv(OUT/"scenario_registry.csv").set_index("Date")
records=[row for path in sorted((OUT/"evaluation").glob("*.json")) for row in json.loads(path.read_text(encoding="utf-8"))]
dates=sorted({r["date"] for r in records})
methods=list(METHODS)+protocol["baseline_methods"]
assert len(records)==len(dates)*len(methods)*len(protocol["evaluation_seeds"])
assert all(r["n_eval"]==lock["budget"] for r in records)
refdata={date:json.loads((OUT/"reference"/f"{date}.json").read_text(encoding="utf-8")) for date in dates}
fronts={};summary_rows=[];front_rows=[];history_rows=[];reference_rows=[]
for date in dates:
    rows=[r for r in records if r["date"]==date]
    candidates=np.vstack([refdata[date]["front_f"],*[r["front_f"] for r in rows]])
    reference=candidates[nondominated_indices(candidates)]
    fronts[date]=reference
    refhv=hypervolume((reference-low)/span)
    for i,f in enumerate(reference):
        reference_rows.append(dict(date=date,point=i,TN_out=f[0],DEC=f[1]))
    for row in rows:
        raw=np.asarray(row["front_f"]);U=np.asarray(row["front_u"])
        case=registry.loc[date].to_dict();case["Date"]=date
        features=feature_matrix(case,U)
        score=dict(date=date,method=row["method"],seed=row["seed"],hv=row["hv"],igd_plus=igd_plus((reference-low)/span,(raw-low)/span),
                   hv_attainment=100*row["hv"]/max(refhv,1e-15),seconds=row["seconds"],n_eval=row["n_eval"],front_size=len(raw),
                   outside_reference_fraction=row["outside_reference_fraction"])
        for ref in protocol["reference_sensitivity"]:
            score[f"hv_ref_{ref}"]=hypervolume((raw-low)/span,(ref,ref))
        summary_rows.append(score)
        for h in row["history"]:
            history_rows.append(dict(date=date,method=row["method"],seed=row["seed"],**h,hv_attainment=100*h["hv"]/max(refhv,1e-15)))
        for j,(u,f) in enumerate(zip(U,raw)):
            front_rows.append(dict(date=date,method=row["method"],seed=row["seed"],u_PPA=u[0],u_DO=u[1],PPA=features.iloc[j].PPA,DO=features.iloc[j].DO,TN_out=f[0],DEC=f[1]))
scores=pd.DataFrame(summary_rows);scores.to_csv(OUT/"evaluation_metrics.csv",index=False)
pd.DataFrame(front_rows).to_csv(OUT/"pareto_points.csv",index=False)
pd.DataFrame(history_rows).to_csv(OUT/"convergence.csv",index=False)
pd.DataFrame(reference_rows).to_csv(OUT/"empirical_reference_front.csv",index=False)
seedmeans=scores.groupby(["method","seed"])[["hv","igd_plus","hv_attainment","seconds","front_size"]].mean().reset_index()
seedmeans.to_csv(OUT/"evaluation_seed_means.csv",index=False)
summary=seedmeans.groupby("method").agg(hv_mean=("hv","mean"),hv_sd=("hv","std"),igd_mean=("igd_plus","mean"),igd_sd=("igd_plus","std"),attainment_mean=("hv_attainment","mean"),attainment_sd=("hv_attainment","std"),seconds_mean=("seconds","mean"),front_size_mean=("front_size","mean")).sort_values("hv_mean",ascending=False)
summary.to_csv(OUT/"method_summary.csv")

# Pair by date and seed, then average by month before the cluster bootstrap.
paired=[];rng=np.random.default_rng(20260905)
for method in METHODS:
    wide=scores[scores.method.isin([method,"UniformRandom"])].pivot(index=["date","seed"],columns="method",values="hv")
    differences=(wide[method]-wide.UniformRandom).rename("delta").reset_index();differences["month"]=differences.date.str[:7]
    monthly=differences.groupby("month").delta.mean().to_numpy()
    boot=rng.choice(monthly,size=(10000,len(monthly)),replace=True).mean(axis=1)
    paired.append(dict(method=method,mean_delta_hv=float(differences.delta.mean()),monthly_equal_weight_delta=float(monthly.mean()),month_cluster_q025=float(np.quantile(boot,.025)),month_cluster_q975=float(np.quantile(boot,.975)),positive_pair_fraction=float((differences.delta>0).mean()),months=len(monthly),date_seed_pairs=len(wide)))
pd.DataFrame(paired).to_csv(OUT/"paired_comparison.csv",index=False)

# Principal method fixed using tuning data only. Five-seed archive union is for
# interpretation; never use this union to replace per-run benchmark scores.
solutions=[];primary=lock["primary_method"]
for date in dates:
    rows=[r for r in records if r["date"]==date and r["method"]==primary]
    F=np.vstack([r["front_f"] for r in rows]);U=np.vstack([r["front_u"] for r in rows]);keep=nondominated_indices(F);F,U=F[keep],U[keep]
    ideal=F.min(axis=0);compromise=int(np.argmin(np.sum(((F-ideal)/span)**2,axis=1)))
    selection={"minimum_TN":int(np.argmin(F[:,0])),"minimum_DEC":int(np.argmin(F[:,1])),"compromise":compromise}
    case=registry.loc[date].to_dict();case["Date"]=date
    X=feature_matrix(case,U);historical=np.array(refdata[date]["historical_proxy"])
    hist_u=np.array(refdata[date]["historical_u"])
    for role,i in selection.items():
        raw=F[i];delta=raw-historical
        solutions.append(dict(date=date,method=primary,role=role,T=case["T"],Q=case["Q"],TN_in=case["TN_in"],MLSS=case["MLSS"],
                              PPA=float(X.iloc[i].PPA),DO=float(X.iloc[i].DO),historical_PPA=case["PPA"],historical_DO=case["DO"],
                              TN_out=raw[0],DEC=raw[1],baseline_TN=historical[0],baseline_DEC=historical[1],
                              delta_TN=delta[0],delta_DEC=delta[1],pct_TN=100*delta[0]/historical[0],pct_DEC=100*delta[1]/historical[1],
                              u_PPA=U[i,0],u_DO=U[i,1],historical_in_search_domain=bool(np.all((hist_u>=0)&(hist_u<=1))),
                              combined_front_size=len(F),front_TN_span=float(np.ptp(F[:,0])),front_DEC_span=float(np.ptp(F[:,1]))))
solutions=pd.DataFrame(solutions);solutions.to_csv(OUT/"representative_solutions.csv",index=False)

residuals=pd.concat([pd.read_csv(OUT/f"{target}_unique_residuals.csv",index_col=0) for target in ["TN_out","DEC"]],axis=1).dropna()
R=residuals.to_numpy();uncertainty=[]
for rho in [0.,.5,.9]:
    # Correlated empirical errors share a component, which cancels in differences.
    eps=np.sqrt(1-rho)*(R[rng.integers(len(R),size=20000)]-R[rng.integers(len(R),size=20000)])
    for _,row in solutions[solutions.role=="compromise"].iterrows():
        perturbed=eps+row[["delta_TN","delta_DEC"]].to_numpy(float)
        lo,hi=np.quantile(perturbed,[.025,.975],axis=0)
        uncertainty.append(dict(date=row.date,correlation=rho,TN_q025=lo[0],TN_q975=hi[0],DEC_q025=lo[1],DEC_q975=hi[1],
                                TN_direction_resolved=bool(lo[0]>0 or hi[0]<0),DEC_direction_resolved=bool(lo[1]>0 or hi[1]<0),
                                interpretation="empirical residual sensitivity, not calibrated counterfactual confidence interval"))
pd.DataFrame(uncertainty).to_csv(OUT/"solution_residual_sensitivity.csv",index=False)
fixed=[]
for date in dates:
    ref=refdata[date];F=np.asarray(ref["fixed_do_front_f"]);U=np.asarray(ref["fixed_do_front_u"]);ideal=F.min(axis=0)
    i=np.argmin(np.sum(((F-ideal)/span)**2,axis=1));base=np.array(ref["historical_proxy"])
    full=solutions[(solutions.date==date)&(solutions.role=="compromise")].iloc[0]
    fixed.append(dict(date=date,fixed_do_historical_supported=ref["fixed_do_is_historical_supported"],fixed_DO=ref["fixed_do_model_coordinate"],
                      fixed_TN=F[i,0],fixed_DEC=F[i,1],fixed_delta_TN=F[i,0]-base[0],fixed_delta_DEC=F[i,1]-base[1],
                      joint_TN=full.TN_out,joint_DEC=full.DEC,joint_minus_fixed_TN=full.TN_out-F[i,0],joint_minus_fixed_DEC=full.DEC-F[i,1]))
pd.DataFrame(fixed).to_csv(OUT/"fixed_do_comparison.csv",index=False)

comp=solutions[solutions.role=="compromise"]
anchors=["2025-07-15","2025-09-15","2025-11-15","2025-12-15"]
representative_dates=[min(dates,key=lambda d:abs((pd.Timestamp(d)-pd.Timestamp(anchor)).days)) for anchor in anchors]
report=dict(primary_method=primary,primary_selection_basis=lock["selection_basis"],evaluation_best_hv=summary.index[0],budget=lock["budget"],
            tuning_cases=int(((pd.read_csv(OUT/"scenario_registry.csv").selected)&(pd.read_csv(OUT/"scenario_registry.csv").period=="tune")).sum()),evaluation_cases=len(dates),
            method_summary=summary.reset_index().to_dict("records"),paired_comparison=paired,representative_dates=representative_dates,
            compromise_means=comp[["TN_out","DEC","baseline_TN","baseline_DEC","delta_TN","delta_DEC","pct_TN","pct_DEC","PPA","DO","front_TN_span","front_DEC_span"]].mean().to_dict(),
            compromise_ranges={c:[float(comp[c].min()),float(comp[c].max())] for c in ["delta_TN","delta_DEC","pct_TN","pct_DEC","PPA","DO"]},
            jointly_improving_cases=int(((comp.delta_TN<0)&(comp.delta_DEC<0)).sum()),historical_in_domain_cases=int(comp.historical_in_search_domain.sum()),
            fixed_do_historically_supported_cases=int(sum(r["fixed_do_historical_supported"] for r in fixed)),
            sensitivity_summary=pd.DataFrame(uncertainty).groupby("correlation")[["TN_direction_resolved","DEC_direction_resolved"]].sum().reset_index().to_dict("records"),
            total_tuning_evaluations=3*6*3*12*1024,total_evaluation_evaluations=len(records)*lock["budget"],
            development_convergence=lock["convergence"],completed_utc=datetime.now(timezone.utc).isoformat())
write_json(OUT/"analysis_summary.json",report)
print(json.dumps(report,ensure_ascii=False,indent=2))
