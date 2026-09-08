"""Post-evaluation domain-matched comparison; does not change benchmark scores."""
from context_moo import *
from datetime import datetime,timezone
R=ROOT/"results"
registry=pd.read_csv(R/"scenario_registry.csv").set_index("Date")
solutions=pd.read_csv(R/"representative_solutions.csv")
dates=sorted(solutions.date.unique());cases=[];X=[];coords=[]
for date in dates:
    case=registry.loc[date].to_dict();case["Date"]=date
    u=np.array([(case[a]-case[f"low_{a}"])/(case[f"high_{a}"]-case[f"low_{a}"]) for a in ACTIONS])
    frame=feature_matrix(case,np.clip(u,0,1)[None,:]);X.append(frame);coords.append(frame.iloc[0][list(ACTIONS)].to_numpy(float))
bundle=load_proxy();F=predict(bundle,pd.concat(X,ignore_index=True))
anchor=pd.DataFrame({"date":dates,"anchor_PPA":[v[0] for v in coords],"anchor_DO":[v[1] for v in coords],"anchor_TN":F[:,0],"anchor_DEC":F[:,1]})
anchor.to_csv(R/"feasible_anchor.csv",index=False)
joined=solutions.merge(anchor,on="date",validate="many_to_one")
for t in ["TN","DEC"]:
    col="TN_out" if t=="TN" else "DEC"
    joined[f"delta_anchor_{t}"]=joined[col]-joined[f"anchor_{t}"]
    joined[f"pct_anchor_{t}"]=100*joined[f"delta_anchor_{t}"]/joined[f"anchor_{t}"]
joined.to_csv(R/"solutions_with_feasible_anchor.csv",index=False)
comp=joined[joined.role=="compromise"]
summary=json.loads((R/"analysis_summary.json").read_text(encoding="utf-8"))
summary["feasible_anchor_diagnostic"]={"role":"post-evaluation same-domain sensitivity, not an algorithm baseline or selection input",
    "created_utc":datetime.now(timezone.utc).isoformat(),"mean":comp[["delta_anchor_TN","delta_anchor_DEC","pct_anchor_TN","pct_anchor_DEC","anchor_TN","anchor_DEC"]].mean().to_dict(),
    "ranges":{c:[float(comp[c].min()),float(comp[c].max())] for c in ["delta_anchor_TN","delta_anchor_DEC"]},
    "jointly_improving_cases":int(((comp.delta_anchor_TN<0)&(comp.delta_anchor_DEC<0)).sum()),
    "anchor_dominates_compromise_cases":int(((comp.delta_anchor_TN>0)&(comp.delta_anchor_DEC>0)).sum())}
unc=pd.read_csv(R/"solution_residual_sensitivity.csv").merge(comp[["date","delta_TN","delta_DEC","delta_anchor_TN","delta_anchor_DEC"]],on="date",validate="many_to_one")
for target in ["TN","DEC"]:
    shift=unc[f"delta_anchor_{target}"]-unc[f"delta_{target}"]
    unc[f"{target}_q025"]+=shift;unc[f"{target}_q975"]+=shift
    unc[f"{target}_direction_resolved"]=(unc[f"{target}_q025"]>0)|(unc[f"{target}_q975"]<0)
unc.to_csv(R/"solution_anchor_residual_sensitivity.csv",index=False)
summary["feasible_anchor_diagnostic"]["sensitivity_summary"]=unc.groupby("correlation")[["TN_direction_resolved","DEC_direction_resolved"]].sum().reset_index().to_dict("records")
write_json(R/"analysis_summary.json",summary)
print(json.dumps(summary["feasible_anchor_diagnostic"],ensure_ascii=False,indent=2))
